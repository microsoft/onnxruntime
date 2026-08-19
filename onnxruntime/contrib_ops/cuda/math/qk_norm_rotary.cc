// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/qk_norm_rotary.h"

#include "contrib_ops/cuda/math/qk_norm_rotary_impl.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      QKNormRotaryEmbedding,                                          \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<float>()), \
      QKNormRotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
QKNormRotaryEmbedding<T>::QKNormRotaryEmbedding(const OpKernelInfo& info) : CudaKernel(info) {
  num_heads_ = info.GetAttrOrDefault<int64_t>("num_heads", static_cast<int64_t>(0));
  head_dim_ = info.GetAttrOrDefault<int64_t>("head_dim", static_cast<int64_t>(0));
  rope_head_dim_ = info.GetAttrOrDefault<int64_t>("rope_head_dim", static_cast<int64_t>(0));
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
  simulate_fp8_ = info.GetAttrOrDefault<int64_t>("simulate_fp8", static_cast<int64_t>(1)) != 0;

  ORT_ENFORCE(num_heads_ > 0, "num_heads must be positive, got ", num_heads_);
  ORT_ENFORCE(head_dim_ > 0, "head_dim must be positive, got ", head_dim_);
  ORT_ENFORCE(rope_head_dim_ >= 0 && rope_head_dim_ <= head_dim_,
              "rope_head_dim must be in [0, head_dim], got ", rope_head_dim_);
  ORT_ENFORCE(rope_head_dim_ % 2 == 0, "rope_head_dim must be even, got ", rope_head_dim_);
  // The activation quantisation groups the non-rotary channels 64 at a time, exactly as the
  // unfused graph reshapes them.
  ORT_ENFORCE(!simulate_fp8_ || (head_dim_ - rope_head_dim_) % 64 == 0,
              "simulate_fp8 needs head_dim - rope_head_dim to be a multiple of 64, got ",
              head_dim_ - rope_head_dim_);
}

template <typename T>
Status QKNormRotaryEmbedding<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* kv = context->Input<Tensor>(1);
  const Tensor* kv_norm_weight = context->Input<Tensor>(2);
  const Tensor* cos_table = context->Input<Tensor>(3);
  const Tensor* sin_table = context->Input<Tensor>(4);

  const auto& q_dims = query->Shape().GetDims();
  if (q_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "query is expected to be [batch, sequence, num_heads * head_dim], "
                           "got rank ",
                           q_dims.size());
  }
  const int64_t batch = q_dims[0];
  const int64_t seq_len = q_dims[1];
  if (q_dims[2] != num_heads_ * head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query must be ",
                           num_heads_ * head_dim_, " wide, got ", q_dims[2]);
  }

  const TensorShape kv_shape({batch, seq_len, head_dim_});
  if (kv->Shape() != kv_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv must be [", batch, ", ", seq_len,
                           ", ", head_dim_, "], got ", kv->Shape());
  }
  if (kv_norm_weight->Shape().Size() != head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv_norm_weight must hold ",
                           head_dim_, " elements, got ", kv_norm_weight->Shape().Size());
  }
  const int64_t rope_elems = batch * seq_len * rope_head_dim_;
  if (rope_head_dim_ > 0 &&
      (cos_table->Shape().Size() != rope_elems || sin_table->Shape().Size() != rope_elems)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the rotary slices must hold ",
                           rope_elems, " elements each.");
  }

  Tensor* query_out = context->Output(0, TensorShape({batch, seq_len, num_heads_, head_dim_}));
  Tensor* kv_out = context->Output(1, kv_shape);
  if (batch * seq_len == 0) return Status::OK();

  QKNormRotaryEmbeddingParams params;
  params.batch = static_cast<int>(batch);
  params.seq_len = static_cast<int>(seq_len);
  params.num_heads = static_cast<int>(num_heads_);
  params.head_dim = static_cast<int>(head_dim_);
  params.rope_head_dim = static_cast<int>(rope_head_dim_);
  params.nope_dim = static_cast<int>(head_dim_ - rope_head_dim_);
  params.epsilon = epsilon_;
  params.simulate_fp8 = simulate_fp8_;

  return LaunchQKNormRotaryEmbedding<T>(Stream(context), params, query->Data<T>(), kv->Data<T>(),
                                        kv_norm_weight->Data<float>(), cos_table->Data<float>(),
                                        sin_table->Data<float>(), query_out->MutableData<T>(),
                                        kv_out->MutableData<T>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
