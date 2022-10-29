// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove GetDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/miopen_common.h"
#include "contrib_ops/rocm/bert/fast_gelu.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info) {
  const TransformerOptions* options = TransformerOptions::GetInstance();
  tuning_ = options->IsTuningEnabled();
}

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, input->Shape());

  int64_t input_length = input->Shape().Size();
  if (input_length == 0) {
    return Status::OK();
  }
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();
  typedef typename ToHipType<T>::MappedType HipT;

  return LaunchFastGeluKernel<HipT>(Stream(),
                                  static_cast<int>(input_length),
                                  static_cast<int>(bias_length),
                                  reinterpret_cast<const HipT*>(input->Data<T>()),
                                  (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->Data<T>()) : nullptr,
                                  reinterpret_cast<HipT*>(output->MutableData<T>()),
                                  tuning_);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
