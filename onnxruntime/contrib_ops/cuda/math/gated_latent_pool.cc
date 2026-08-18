// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/gated_latent_pool.h"

#include "contrib_ops/cuda/math/gated_latent_pool_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      GatedLatentPool,                                                                  \
      kMSDomain,                                                                        \
      1,                                                                                \
      T,                                                                                \
      kCudaExecutionProvider,                                                           \
      (*KernelDefBuilder::Create())                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                        \
          .TypeConstraint("P", BuildKernelDefConstraints<float, MLFloat16, BFloat16>()) \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<float>())                    \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),                 \
      GatedLatentPool<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
GatedLatentPool<T>::GatedLatentPool(const OpKernelInfo& info) : CudaKernel(info) {
  ratio_ = info.GetAttrOrDefault<int64_t>("ratio", static_cast<int64_t>(4));
  window_multiplier_ = info.GetAttrOrDefault<int64_t>("window_multiplier", static_cast<int64_t>(1));
  head_dim_ = info.GetAttrOrDefault<int64_t>("head_dim", static_cast<int64_t>(0));
  rope_head_dim_ = info.GetAttrOrDefault<int64_t>("rope_head_dim", static_cast<int64_t>(0));
  max_seq_len_ = info.GetAttrOrDefault<int64_t>("max_seq_len", static_cast<int64_t>(0));
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
  simulate_fp8_ = info.GetAttrOrDefault<int64_t>("simulate_fp8", static_cast<int64_t>(0)) != 0;
  simulate_rotated_fp4_ = info.GetAttrOrDefault<int64_t>("simulate_rotated_fp4", static_cast<int64_t>(0)) != 0;

  ORT_ENFORCE(ratio_ >= 1, "ratio must be positive, got ", ratio_);
  ORT_ENFORCE(window_multiplier_ == 1 || window_multiplier_ == 2, "window_multiplier must be 1 or 2, got ", window_multiplier_);
  ORT_ENFORCE(head_dim_ > 0, "head_dim must be positive, got ", head_dim_);
  ORT_ENFORCE(rope_head_dim_ >= 0 && rope_head_dim_ <= head_dim_ && rope_head_dim_ % 2 == 0,
              "rope_head_dim must be even and in [0, head_dim], got ", rope_head_dim_);
  ORT_ENFORCE(max_seq_len_ > 0, "max_seq_len must be positive, got ", max_seq_len_);
  ORT_ENFORCE(!simulate_fp8_ || (head_dim_ - rope_head_dim_) % 64 == 0,
              "simulate_fp8 needs head_dim - rope_head_dim to be a multiple of 64, got ",
              head_dim_ - rope_head_dim_);
  // The rotation is a Walsh-Hadamard butterfly, and the FP4 grouping is 32 wide.
  ORT_ENFORCE(!simulate_rotated_fp4_ || ((head_dim_ & (head_dim_ - 1)) == 0 && head_dim_ % 32 == 0),
              "simulate_rotated_fp4 needs head_dim to be a power of two of at least 32, got ", head_dim_);
}

template <typename T>
Status GatedLatentPool<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* kv = context->Input<Tensor>(0);
  const Tensor* score = context->Input<Tensor>(1);
  const Tensor* past_state_kv = context->Input<Tensor>(2);
  const Tensor* past_state_score = context->Input<Tensor>(3);
  const Tensor* ape = context->Input<Tensor>(4);
  const Tensor* norm_weight = context->Input<Tensor>(5);
  const Tensor* cos_table = context->Input<Tensor>(6);
  const Tensor* sin_table = context->Input<Tensor>(7);
  const Tensor* past_lens = context->Input<Tensor>(8);

  const auto& kv_dims = kv->Shape().GetDims();
  if (kv_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "kv is expected to be [batch, sequence, window_multiplier * head_dim], got rank ",
                           kv_dims.size());
  }

  const int64_t batch = kv_dims[0];
  const int64_t seq_len = kv_dims[1];
  const int64_t feat = window_multiplier_ * head_dim_;
  const int64_t span = window_multiplier_ * ratio_;
  const int64_t num_rows = (seq_len - 1) / ratio_ + 2;

  // With `score` omitted the two projections came out of one GEMM, so kv is twice as wide and
  // holds them side by side.  Nothing downstream needs them separated: the kernel only ever
  // indexes a row, and a wider stride is all it takes to keep reading the same values.
  const bool fused = score == nullptr;
  const int64_t proj_feat = fused ? 2 * feat : feat;
  if (kv_dims[2] != proj_feat) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv must be ", proj_feat,
                           " wide, got ", kv_dims[2]);
  }
  if (!fused) {
    if (score->Shape() != kv->Shape()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "score must have the same shape as kv.");
    }
    if (score->DataType() != kv->DataType()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "kv and score must have the same element type.");
    }
  }
  const TensorShape state_shape({batch, span, feat});
  if (past_state_kv->Shape() != state_shape || past_state_score->Shape() != state_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the rolling state must be shaped [",
                           batch, ", ", span, ", ", feat, "].");
  }
  if (ape->Shape().Size() != ratio_ * feat) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ape must hold ", ratio_ * feat,
                           " elements, got ", ape->Shape().Size());
  }
  if (norm_weight->Shape().Size() != head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "norm_weight must hold ", head_dim_,
                           " elements, got ", norm_weight->Shape().Size());
  }
  const TensorShape rope_shape({max_seq_len_, rope_head_dim_});
  if (rope_head_dim_ > 0 && (cos_table->Shape() != rope_shape || sin_table->Shape() != rope_shape)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the rotary tables must be shaped [",
                           max_seq_len_, ", ", rope_head_dim_, "].");
  }
  if (past_lens->Shape().Size() != batch) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "past_lens must hold ", batch,
                           " elements, got ", past_lens->Shape().Size());
  }

  Tensor* rows = context->Output(0, TensorShape({batch, num_rows, head_dim_}));
  Tensor* first_slot = context->Output(1, TensorShape({batch, 1}));
  Tensor* last_slot = context->Output(2, TensorShape({batch, 1}));
  Tensor* row_count = context->Output(3, TensorShape({}));
  Tensor* present_state_kv = context->Output(4, state_shape);
  Tensor* present_state_score = context->Output(5, state_shape);

  GatedLatentPoolParams params;
  params.batch = static_cast<int>(batch);
  params.seq_len = static_cast<int>(seq_len);
  params.num_rows = static_cast<int>(num_rows);
  params.ratio = static_cast<int>(ratio_);
  params.window_multiplier = static_cast<int>(window_multiplier_);
  params.span = static_cast<int>(span);
  params.head_dim = static_cast<int>(head_dim_);
  params.rope_head_dim = static_cast<int>(rope_head_dim_);
  params.nope_dim = static_cast<int>(head_dim_ - rope_head_dim_);
  params.feat = static_cast<int>(feat);
  params.proj_feat = static_cast<int>(proj_feat);
  params.max_seq_len = static_cast<int>(max_seq_len_);
  params.epsilon = epsilon_;
  params.simulate_fp8 = simulate_fp8_;
  params.simulate_rotated_fp4 = simulate_rotated_fp4_;

  // The projections carry no type constraint of their own in the kernel registration, so the
  // element type is whatever the producing MatMul emitted and has to be picked up here.
  auto launch = [&](auto proj) {
    using P = decltype(proj);
    const P* kv_data = kv->Data<P>();
    return LaunchGatedLatentPool<T, P>(Stream(context), params,
                                       kv_data,
                                       fused ? kv_data + feat : score->Data<P>(),
                                       past_state_kv->Data<float>(),
                                       past_state_score->Data<float>(),
                                       ape->Data<float>(),
                                       norm_weight->Data<float>(),
                                       cos_table->Data<float>(),
                                       sin_table->Data<float>(),
                                       past_lens->Data<int64_t>(),
                                       rows->MutableData<T>(),
                                       first_slot->MutableData<int64_t>(),
                                       last_slot->MutableData<int64_t>(),
                                       row_count->MutableData<int64_t>(),
                                       present_state_kv->MutableData<float>(),
                                       present_state_score->MutableData<float>());
  };

  if (kv->IsDataType<float>()) return launch(float{});
  if (kv->IsDataType<MLFloat16>()) return launch(MLFloat16{});
  if (kv->IsDataType<BFloat16>()) return launch(BFloat16{});
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "kv and score must be float, float16 or bfloat16.");
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
