// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/shrink.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/math.h"
#include "core/framework/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0,
    element_type_lists::AllNumeric);
}

using EnabledShrinkDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0);

ONNX_CPU_OPERATOR_KERNEL(
    Shrink,
    9,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledShrinkDataTypes>()),
    Shrink);
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif
namespace shrink_internal {
template <class T>
inline T ShrinkCore(const T& t_val, float bias, float lambd) {
  // The ONNX spec doesn't take numeric overflow and underflow into account
  // Implementing the spec as is for now
  float val = static_cast<float>(t_val);
  if (val < -lambd) {
    return static_cast<T>(val + bias);
  }
  if (val > lambd) {
    return static_cast<T>(val - bias);
  } else {
    return static_cast<T>(0.f);
  }
}

template <class T>
struct CallShrinkImpl {
  Status operator()(const Tensor* input, Tensor* output, float bias, float lambd) const {
    EigenMap<T>(*output) = EigenMap<T>(*input).unaryExpr([bias, lambd](const T& val) { return ShrinkCore<T>(val, bias, lambd); });
    return Status::OK();
  }
};

}  // namespace shrink_internal

Status Shrink::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  auto* output = p_op_kernel_context->Output(0, input->Shape());
  utils::MLTypeCallDispatcherFromTypeList<EnabledShrinkDataTypes> t_disp(input->GetElementType());
  return t_disp.InvokeRet<Status, shrink_internal::CallShrinkImpl>(input, output, bias_, lambd_);
}
}  // namespace onnxruntime
