// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "common.h"
#include "adam.h"

namespace onnxruntime {
namespace cuda {

// TODO: Once Schema is checked in to onnx lets fix this to match that
#define REGISTER_ADAM_KERNEL_TYPED(T1, T2, T3, T4, T_GRAD)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                      \
      AdamOptimizer,                                                                  \
      kOnnxDomain,                                                                    \
      9,                                                                              \
      T1##_##T2##_##T3##_##T4##_##T_GRAD,                                             \
      kCudaExecutionProvider,                                                         \
      KernelDefBuilder()                                                              \
          .Alias(1, 3)                             /* Update step count in-place */   \
          .Alias(2, 0)                             /* Update weights in-place */      \
          .Alias(4, 1)                             /* Update moment-1 in-place */     \
          .Alias(5, 2)                             /* Update moment-2 in-place */     \
          .Alias(6, 4)                             /* Update FP16 weights in-place */ \
          .InputMemoryType<OrtMemTypeCPUInput>(1)  /* Keep step count in CPU */       \
          .InputMemoryType<OrtMemTypeCPUInput>(8)  /* Keep do_update in CPU */        \
          .OutputMemoryType<OrtMemTypeCPUInput>(3) /* Keep step count in CPU */       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                    \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())                    \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())                    \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>())            \
          .TypeConstraint("T_FP16", DataTypeImpl::GetTensorType<MLFloat16>())         \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),                  \
      AdamOptimizer<T1, T2, T3, T4, T_GRAD>);

REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, float)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, MLFloat16)

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
Status AdamOptimizer<T1, T2, T3, T4, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NW = *ctx->Output(0, W.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor& NS = *ctx->Output(3, S.Shape());

  half* fp16_weights_out = nullptr;
  if (ctx->InputCount() >= 7 && ctx->OutputCount() >= 5) {
    const Tensor& W_FP16 = *ctx->Input<Tensor>(6);
    Tensor& NW_FP16 = *ctx->Output(4, W_FP16.Shape());
    fp16_weights_out = reinterpret_cast<half*>(NW_FP16.template MutableData<MLFloat16>());
  }

  const CudaT3* loss_scale = nullptr;
  if (ctx->InputCount() >= 8) {
    const Tensor& loss_scale_tensor = *ctx->Input<Tensor>(7);
    loss_scale = reinterpret_cast<const CudaT3*>(loss_scale_tensor.template Data<T3>());
  }

  const T2* S_in = S.template Data<T2>();
  if (ctx->InputCount() >= 9) {
    const Tensor& do_update_tensor = *ctx->Input<Tensor>(8);
    const bool do_update = *do_update_tensor.template Data<bool>();
    if (!do_update) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T3>(W, NW));
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(M1, NM1));
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(M2, NM2));
      if (S_in != NS.template MutableData<T2>()) {
        *(NS.template MutableData<T2>()) = *(S_in);
      }

      if (fp16_weights_out) {
        const Tensor& W_FP16 = *ctx->Input<Tensor>(6);
        Tensor& NW_FP16 = *ctx->Output(4, W_FP16.Shape());
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<MLFloat16>(W_FP16, NW_FP16));
      }
      return Status::OK();
    }
  }

  AdamOptimizerImpl(
      reinterpret_cast<const CudaT1*>(ETA.template Data<T1>()),
      *S_in,
      reinterpret_cast<const CudaT3*>(W.template Data<T3>()),
      reinterpret_cast<const CudaT_GRAD*>(G.template Data<T_GRAD>()),
      reinterpret_cast<const CudaT4*>(M1.template Data<T4>()),
      reinterpret_cast<const CudaT4*>(M2.template Data<T4>()),
      loss_scale,
      ToCudaType<T4>::FromFloat(alpha_),
      ToCudaType<T4>::FromFloat(beta_),
      ToCudaType<T4>::FromFloat(lambda_),
      ToCudaType<T4>::FromFloat(epsilon_),
      reinterpret_cast<CudaT3*>(NW.template MutableData<T3>()),
      reinterpret_cast<CudaT4*>(NM1.template MutableData<T4>()),
      reinterpret_cast<CudaT4*>(NM2.template MutableData<T4>()),
      fp16_weights_out,
      W.Shape().Size());

  *(NS.template MutableData<T2>()) = *(S_in) + 1;

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
