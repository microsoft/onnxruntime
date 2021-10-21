// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_control.h"

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {
ONNX_OPERATOR_KERNEL_EX(
    InPlaceAccumulator,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)  // accumulate tensors in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    InPlaceAccumulator<float>);

template <typename T>
Status InPlaceAccumulator<T>::Compute(OpKernelContext* context) const {
  const Tensor* gradient_buffer = context->Input<Tensor>(0);
  const Tensor* do_update_tensor = context->Input<Tensor>(2);
  Tensor* accumulated_gradient = context->Output(0, gradient_buffer->Shape());
  const void* input_data = gradient_buffer->template Data<T>();
  void* output_data = accumulated_gradient->template MutableData<T>();
  if (do_update_tensor) {
    const bool do_update = *(do_update_tensor->template Data<bool>());
    if (!do_update) {
      if (output_data != input_data) {
        memcpy(output_data, input_data, gradient_buffer->SizeInBytes());
      }
      return Status::OK();
    }
  }

  //Copy from Add CPU kernel
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() + per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() + per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>() + per_iter_bh.EigenInput1<T>();
      }};

  UntypedBroadcastTwo(*context, funcs);

  return Status::OK();
}

template <typename T>
Status ZeroGradient<T>::Compute(OpKernelContext* context) const {
  const Tensor& old_gradient = *context->Input<Tensor>(0);
  Tensor& zero_gradient = *context->Output(0, old_gradient.Shape());

  std::memset(zero_gradient.template MutableData<T>(), 0, zero_gradient.Shape().Size() * sizeof(T));
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    ZeroGradient,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)  // reset gradients in-place
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::AllTensorTypes()),
    ZeroGradient<float>);

}  // namespace contrib
}  // namespace onnxruntime
