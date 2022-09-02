// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/gemm_fast_gelu.h"

#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "contrib_ops/rocm/bert/transformer_common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/rocm/math/matmul_impl.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using onnxruntime::rocm::MatMulImpl;
using onnxruntime::rocm::ToHipType;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GemmFastGelu,                                               \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GemmFastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status GemmFastGelu<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<T>::MappedType HipT;

  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const auto* bias = ctx->Input<Tensor>(2);

  bool transa = false;
  bool transb = false;
  bool trans_batch_a = false;
  bool trans_batch_b = false;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(X->Shape(), W->Shape(), transa, transb, trans_batch_a, trans_batch_b, false));

  auto gemm_buffer = GetScratchBuffer<T>(helper.OutputShape().Size());
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const float alpha = 1.0f;
  const float zero = 0.0f;

  if (MatMulImpl<T>(this, helper, reinterpret_cast<const T*>(X->Data<T>()),
                    reinterpret_cast<const T*>(W->Data<T>()),
                    reinterpret_cast<T*>(gemm_buffer.get()),
                    X->Shape(), W->Shape(),
                    transa, transb, trans_batch_a, trans_batch_b, alpha, zero) != Status::OK()) {
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  int64_t fast_gelu_input_length = Y->Shape().Size();
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();

  return LaunchFastGeluKernel<HipT>(Stream(),
                                  static_cast<int>(fast_gelu_input_length),
                                  static_cast<int>(bias_length),
                                  reinterpret_cast<HipT*>(gemm_buffer.get()),
                                  (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->Data<T>()) : nullptr,
                                  reinterpret_cast<HipT*>(Y->MutableData<T>()),
                                  false);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
