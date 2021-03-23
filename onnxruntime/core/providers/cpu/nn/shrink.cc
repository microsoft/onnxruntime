// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/shrink.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math_cpuonly.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0,
    element_type_lists::AllNumeric);
}

using ShrinkDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0);
using EnabledShrinkDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Shrink, Input, 0);

ONNX_CPU_OPERATOR_KERNEL(
    Shrink,
    9,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<ShrinkDataTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledShrinkDataTypes>()),
    Shrink);

namespace shrink_internal {
template <class T>
inline T ShrinkCore(const T& val, float bias, float lambd) {
  // The ONNX spec doesn't take numeric overflow and underflow into account
  // Implementing the spec as is for now
  if (val < -lambd) {
    return T(val + bias);
  }
  if (val > lambd) {
    return T(val - bias);
  } else {
    return T(0);
  }
}

template <class T>
Status ShrinkImpl(const Tensor* input, Tensor* output, float bias, float lambd) {
  EigenMap<T>(*output) = EigenMap<T>(*input).unaryExpr([bias, lambd](const T& val) { return ShrinkCore<T>(val, bias, lambd); });
  return Status::OK();
}

template <>
Status ShrinkImpl<MLFloat16>(const Tensor* input, Tensor* output, float bias, float lambd) {
  const auto& span = gsl::make_span(input->Data<MLFloat16>(), input->Shape().Size());
  auto* output_data = output->template MutableData<MLFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [bias, lambd](const MLFloat16& val) {
    float fl = math::halfToFloat(val.val);
    return MLFloat16(math::floatToHalf(ShrinkCore<float>(fl, bias, lambd)));
  });
  return Status::OK();
}

template <>
Status ShrinkImpl<BFloat16>(const Tensor* input, Tensor* output, float bias, float lambd) {
  const auto& span = gsl::make_span(input->Data<BFloat16>(), input->Shape().Size());
  auto* output_data = output->template MutableData<BFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [bias, lambd](const BFloat16& val) {
    float fl = val.ToFloat();
    return BFloat16(ShrinkCore<float>(fl, bias, lambd));
  });
  return Status::OK();
}

template <class T>
struct CallShrinkImpl {
  Status operator()(const Tensor* input, Tensor* output, float bias, float lambd) const {
    return ShrinkImpl<T>(input, output, bias, lambd);
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
