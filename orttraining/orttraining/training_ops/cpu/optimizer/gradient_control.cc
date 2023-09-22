// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_control.h"

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
void getBroadcastSpanFunc(ProcessBroadcastSpanFuncs& funcs) {
  ProcessBroadcastSpanFuncs add_funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() + per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() + per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>() + per_iter_bh.EigenInput1<T>();
      }};
  funcs = std::move(add_funcs);
}
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
  void* output_data = accumulated_gradient->template MutableData<T>();
  if (do_update_tensor) {
    const bool do_update = *(do_update_tensor->template Data<bool>());
    if (!do_update) {
      const void* input_data = gradient_buffer->template Data<T>();
      if (output_data != input_data) {
        memcpy(output_data, input_data, gradient_buffer->SizeInBytes());
      }
      return Status::OK();
    }
  }

  // Copy from Add CPU kernel
  ProcessBroadcastSpanFuncs funcs;
  getBroadcastSpanFunc<T>(funcs);

  UntypedBroadcastTwo(*context, funcs);

  return Status::OK();
}

template <typename T>
Status ZeroGradient<T>::Compute(OpKernelContext* context) const {
  const Tensor& old_gradient = *context->Input<Tensor>(0);
  Tensor& zero_gradient = *context->Output(0, old_gradient.Shape());

  std::memset(zero_gradient.template MutableData<T>(), 0, zero_gradient.SizeInBytes());
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

ONNX_OPERATOR_KERNEL_EX(
    InPlaceAccumulatorV2,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 1)  // accumulate tensors in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    InPlaceAccumulatorV2<float>);

template <typename T>
Status InPlaceAccumulatorV2<T>::Compute(OpKernelContext* context) const {
  Tensor* accumulation_buffer = const_cast<Tensor*>(context->Input<Tensor>(0));
  const Tensor* new_value = context->Input<Tensor>(1);
  const Tensor* overwrite_tensor = context->Input<Tensor>(2);

  void* accumulation_buffer_data = accumulation_buffer->template MutableData<T>();
  const bool overwrite = overwrite_tensor != nullptr ? *(overwrite_tensor->template Data<bool>()) : false;

  if (overwrite) {
    const void* updated_data = new_value->template Data<T>();
    memcpy(accumulation_buffer_data, updated_data, new_value->SizeInBytes());
  } else {
    // Copy from Add CPU kernel
    ProcessBroadcastSpanFuncs funcs;
    getBroadcastSpanFunc<T>(funcs);

    InputBroadcaster input_broadcaster(*accumulation_buffer, *new_value);
    OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(), *accumulation_buffer);
    BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster, nullptr);

    BroadcastLooper(broadcast_helper, funcs);
  }

  Tensor* updated_output = context->Output(0, {1});
  bool* updated_output_ptr = updated_output->template MutableData<bool>();
  *updated_output_ptr = true;

  Tensor* accumulated_value_out = context->Output(1, new_value->Shape());
  if (nullptr != accumulated_value_out) {
    void* output_data = accumulated_value_out->template MutableData<T>();
    if (output_data != accumulation_buffer_data) {
      memcpy(output_data, accumulation_buffer_data, new_value->SizeInBytes());
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
