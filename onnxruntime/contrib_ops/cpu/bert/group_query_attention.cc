// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "group_query_attention.h"
#include "group_query_attention_helper.h"

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

// TODO: get this right
// TODO: How can I specify float32 for cpu only
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                         \
ONNX_OPERATOR_TYPED_KERNEL_EX( \
    GroupQueryAttention,   \
    kMSDomain, \
    1, \
    T, \
    kCpuExecutionProvider, \
    KernelDefBuilder() \
        .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
        .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()}), \
    GroupQueryAttention<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info) : OpKernel(info), GQAAttentionBase(info, false) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
}

// Reshape Q/K/V from BxSxD to BxSxNxH
Status Reshape_BSD_to_BSNH(Tensor* qkv,
                           int batch_size,
                           int sequence_length,
                           int num_heads,
                           int head_size) {
  std::vector<int64_t> reshape_dims({batch_size, sequence_length, num_heads, head_size});
  gsl::span<const int64_t> reshape_dims_span{reshape_dims};
  TensorShape qkv_bsnh(reshape_dims_span);
  qkv->Reshape(qkv_bsnh);
  return Status::OK();
}

// Transpose Q/K/V from BxSxNxH to BxNxSxH
Status Transpose_BSNH_to_BNSH(const Tensor* qkv,
                              OrtValue& qkv_transposed,
                              concurrency::ThreadPool* tp = nullptr) {
  std::vector<size_t> permutations({0, 2, 1, 3});
  gsl::span<const size_t> permutations_span{permutations};
  size_t from = 2, to = 1;
  SingleAxisTranspose(permutations_span, *qkv, *qkv_transposed.GetMutable<Tensor>(), from, to, nullptr, tp);
  return Status::OK();
}

template <typename T>
Status MaybeTransposeToBNSH(OpKernelContext* context, AllocatorPtr allocator,
                                      int batch_size, int num_heads, int sequence_length, int head_size,
                                      const Tensor* in, OrtValue& out) {
  auto element_type = DataTypeImpl::GetType<T>();
  std::vector<int64_t> new_dims({batch_size, num_heads, sequence_length, head_size});
  gsl::span<const int64_t> new_dims_span{new_dims};
  TensorShape v_BNLH(new_dims_span);
  Tensor::InitOrtValue(element_type, v_BNLH, allocator, out);
  std::unique_ptr<Tensor> reshaped;
  if (in->Shape().GetDims().size() == 3) {
    reshaped = std::make_unique<Tensor>(in->DataType(), in->Shape(), const_cast<void*>(in->DataRaw()), in->Location());
    ORT_RETURN_IF_ERROR(Reshape_BSD_to_BSNH(reshaped.get(), batch_size, sequence_length, num_heads, head_size));
  }
  ORT_RETURN_IF_ERROR(Transpose_BSNH_to_BNSH((reshaped == nullptr) ? in : reshaped.get(), out));

  return Status::OK();
};

template <typename T>
Status GroupQueryAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* seqlens_k = context->Input<Tensor>(5);
  const Tensor* total_seqlen = context->Input<Tensor>(6);
  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);

  // if (query->Shape().GetDims().size() == 5) {
  //   ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for CPU");
  // }
  // if (key != nullptr && key->Shape().GetDims().size() == 5) {
  //   ORT_NOT_IMPLEMENTED("Packed KV not implemented for CPU");
  // }

  GroupQueryAttentionParameters parameters = {};
  constexpr float scale = 1.0f;
  // bool past_present_share_buffer = false;
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
                                                                total_seqlen,
                                                                /*is_past_bsnh_*/ false,
                                                                scale));

  // TODO: figure out parameters
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  // const int past_kv_seqlen = parameters.seqlen_past_kv_cache;
  const int present_kv_seqlen = parameters.seqlen_present_kv_cache;
  int head_size = parameters.head_size;
  int q_hidden_size = parameters.hidden_size;
  int kv_hidden_size = parameters.kv_hidden_size;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(q_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_k_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(head_size)});
  std::vector<int64_t> present_v_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(head_size)});
  Tensor* present_k = context->Output(1, present_k_shape);
  Tensor* present_v = context->Output(2, present_v_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // TODO: update this comment
  // For each of Q/K/V, there are multiple scenarios:
  // 1) Combined QKV bias is null
  //    a) Q/K/V is (B, S, D)
  //    b) Q/K/V is (B, S, N, H)
  // 2) No packed QKV in Q
  //    a) Q/K/V has seq_len = 1
  //    b) Q/K/V has seq_len > 1

  // TODO: what's with the maybe?
  // TODO: account for packed qkv
  // TODO: make kernel take in BxSxNxH
  OrtValue Q;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
      context, allocator, batch_size, num_heads_, sequence_length, head_size, query, Q));

  OrtValue K;
  OrtValue V;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
      context, allocator, batch_size, kv_num_heads_, sequence_length, head_size, key, K));
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
      context, allocator, batch_size, kv_num_heads_, sequence_length, head_size, value, V));

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(), K.GetMutable<Tensor>()->MutableData<T>(),
                        V.GetMutable<Tensor>()->MutableData<T>(), past_key, past_value, output, present_k, present_v,
                        batch_size, sequence_length, head_size, kv_hidden_size, context);
}
}  // namespace contrib
}  // namespace onnxruntime
