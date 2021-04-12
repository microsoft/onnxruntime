// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "orttraining/training_ops/cuda/optimizer/common.h"
#include "orttraining/training_ops/cuda/optimizer/adam.h"

namespace onnxruntime {
namespace cuda {

// TODO: Once Schema is checked in to onnx lets fix this to match that
#define REGISTER_ADAM_KERNEL_TYPED(T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, T_MIXED_PRECISION_FP)          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                       \
      AdamOptimizer,                                                                                   \
      kMSDomain,                                                                                       \
      1,                                                                                               \
      T1##_##T2##_##T3##_##T4##_##T_GRAD##_##T_GRAD_NORM##_##T_MIXED_PRECISION_FP,                     \
      kCudaExecutionProvider,                                                                          \
      (*KernelDefBuilder::Create())                                                                    \
          .Alias(1, 0)                              /* Update step count in-place */                   \
          .Alias(2, 3)                              /* Update weights in-place */                      \
          .Alias(3, 4)                              /* Update gradients in-place */                    \
          .Alias(4, 1)                              /* Update moment-1 in-place */                     \
          .Alias(5, 2)                              /* Update moment-2 in-place */                     \
          .Alias(6, 5)                              /* Update mixed_precision weights in-place */      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)   /* Keep step count in CPU */                       \
          .InputMemoryType(OrtMemTypeCPUInput, 9)   /* Keep do_update in CPU */                        \
          .OutputMemoryType(OrtMemTypeCPUOutput, 0) /* Keep step count in CPU */                       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                                     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                                     \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())                                     \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())                                     \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>())                             \
          .TypeConstraint("T_MIXED_PRECISION_FP", DataTypeImpl::GetTensorType<T_MIXED_PRECISION_FP>()) \
          .TypeConstraint("T_GRAD_NORM", DataTypeImpl::GetTensorType<T_GRAD_NORM>()),                  \
      AdamOptimizer<T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, T_MIXED_PRECISION_FP>);

REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, float, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, float, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, float, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, MLFloat16, MLFloat16, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, MLFloat16, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, MLFloat16, MLFloat16, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, MLFloat16, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, MLFloat16, MLFloat16, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, MLFloat16, float, MLFloat16)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, float, float, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(BFloat16, int64_t, float, BFloat16, float, float, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, BFloat16, float, float, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, BFloat16, BFloat16, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, BFloat16, float, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(BFloat16, int64_t, float, BFloat16, BFloat16, BFloat16, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(BFloat16, int64_t, float, BFloat16, BFloat16, float, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, BFloat16, BFloat16, BFloat16, BFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, BFloat16, BFloat16, float, BFloat16)
#endif

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD, typename T_GRAD_NORM, typename T_MIXED_PRECISION_FP>
Status AdamOptimizer<T1, T2, T3, T4, T_GRAD, T_GRAD_NORM, T_MIXED_PRECISION_FP>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;
  typedef typename ToCudaType<T_GRAD_NORM>::MappedType CudaT_GRAD_NORM;
  typedef typename ToCudaType<T_MIXED_PRECISION_FP>::MappedType CudaT_MIXED_PRECISION_FP;

  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);
  const Tensor* W_MIXED_FP = ctx->Input<Tensor>(6);
  const Tensor* loss_scale_tensor = ctx->Input<Tensor>(7);
  const Tensor* gradient_norm_tensor = ctx->Input<Tensor>(8);
  const Tensor* do_update_tensor = ctx->Input<Tensor>(9);

  Tensor& NS = *ctx->Output(0, S.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor* NW = ctx->Output(3, W.Shape());
  Tensor* NG = ctx->Output(4, G.Shape());
  Tensor* NW_MIXED_FP = W_MIXED_FP != nullptr ? ctx->Output(5, W_MIXED_FP->Shape()) : nullptr;

  // TODO: temporary hack until View is improved (it doesn't work with Alias)
  if (NW != nullptr)
    NW->SetByteOffset(W.ByteOffset());
  if (NG != nullptr)
    NG->SetByteOffset(G.ByteOffset());
  if (NW_MIXED_FP != nullptr)
    NW_MIXED_FP->SetByteOffset(W_MIXED_FP->ByteOffset());

  CudaT_MIXED_PRECISION_FP* mixed_precision_weights_out = nullptr;
  if (NW_MIXED_FP != nullptr) {
    mixed_precision_weights_out = reinterpret_cast<CudaT_MIXED_PRECISION_FP*>(NW_MIXED_FP->template MutableData<T_MIXED_PRECISION_FP>());
  }

  const CudaT3* loss_scale = nullptr;
  if (loss_scale_tensor != nullptr) {
    loss_scale = reinterpret_cast<const CudaT3*>(loss_scale_tensor->template Data<T3>());
  }

  const T2* S_in = S.template Data<T2>();
  T2* S_out = NS.template MutableData<T2>();

  const CudaT_GRAD_NORM* G_norm = nullptr;
  if (gradient_norm_tensor != nullptr) {
    G_norm = reinterpret_cast<const CudaT_GRAD_NORM*>(gradient_norm_tensor->template Data<T_GRAD_NORM>());
  }

  if (do_update_tensor != nullptr) {
    const bool do_update = *(do_update_tensor->template Data<bool>());
    if (!do_update) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(Stream(), M1, NM1));
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(Stream(), M2, NM2));

      if (S_in != S_out) {
        *(S_out) = *(S_in);
      }
      if (NW != nullptr) {
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T3>(Stream(), W, *NW));
      }
      if (NG != nullptr) {
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T_GRAD>(Stream(), G, *NG));
      }
      if (W_MIXED_FP != nullptr && NW_MIXED_FP != nullptr) {
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T_MIXED_PRECISION_FP>(Stream(), *W_MIXED_FP, *NW_MIXED_FP));
      }

      return Status::OK();
    }
  }

  AdamOptimizerImpl(
      Stream(),
      reinterpret_cast<const CudaT1*>(ETA.template Data<T1>()),
      *S_in,
      reinterpret_cast<const CudaT3*>(W.template Data<T3>()),
      reinterpret_cast<const CudaT_GRAD*>(G.template Data<T_GRAD>()),
      reinterpret_cast<const CudaT4*>(M1.template Data<T4>()),
      reinterpret_cast<const CudaT4*>(M2.template Data<T4>()),
      loss_scale,
      G_norm,
      ToCudaType<T4>::FromFloat(alpha_),
      ToCudaType<T4>::FromFloat(beta_),
      ToCudaType<T4>::FromFloat(lambda_),
      ToCudaType<T4>::FromFloat(epsilon_),
      ToCudaType<T4>::FromFloat(max_norm_clip_),
      do_bias_correction_,
      weight_decay_mode_,
      reinterpret_cast<CudaT4*>(NM1.template MutableData<T4>()),
      reinterpret_cast<CudaT4*>(NM2.template MutableData<T4>()),
      NW != nullptr ? reinterpret_cast<CudaT3*>(NW->template MutableData<T3>()) : nullptr,
      NG != nullptr ? reinterpret_cast<CudaT_GRAD*>(NG->template MutableData<T_GRAD>()) : nullptr,
      mixed_precision_weights_out,
      W.Shape().Size());

  *(S_out) = *(S_in) + 1;

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
