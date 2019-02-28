// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/shrink.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/utils.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    Shrink,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllNumericTensorTypes()),
    Shrink);

namespace shrink_internal {
template <class T>
inline T ShrinkCore(const T& val, float bias, float lambd) {
  // The ONNX spec doesn't take numeric overflow and underflow into account
  // Implementing the spec as is for now
  if (val < -lambd) {
    return T(val + bias);
  } else if (val > lambd) {
    return T(val - bias);
  } else {
    return T(0);
  }
}

template <class T>
void ShrinkImpl(const Tensor* input, Tensor* output, float bias, float lambd) {
  EigenMap<T>(*output) = EigenMap<T>(*input).unaryExpr([bias, lambd](const T& val) { return ShrinkCore<T>(val, bias, lambd); });
}

template <>
void ShrinkImpl<MLFloat16>(const Tensor* input, Tensor* output, float bias, float lambd) {
  const auto& span = gsl::make_span(input->Data<MLFloat16>(), input->Shape().Size());
  auto* output_data = output->template MutableData<MLFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [bias, lambd](const MLFloat16& val) {
    float fl = math::halfToFloat(val.val);
    return MLFloat16(math::floatToHalf(ShrinkCore<float>(fl, bias, lambd)));
  });
}

template <>
void ShrinkImpl<BFloat16>(const Tensor* input, Tensor* output, float bias, float lambd) {
  const auto& span = gsl::make_span(input->Data<BFloat16>(), input->Shape().Size());
  auto* output_data = output->template MutableData<BFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [bias, lambd](const BFloat16& val) {
    float fl = val.ToFloat();
    return BFloat16(ShrinkCore<float>(fl, bias, lambd));
  });
}

}  // namespace shrink_internal

Status Shrink::Compute(OpKernelContext* p_op_kernel_context) const {
  using namespace shrink_internal;

  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  auto* output = p_op_kernel_context->Output(0, input->Shape());
  const auto& dtype = input->DataType();
  DispatchOnNumericTensorType(dtype, ShrinkImpl, input, output, bias_, lambd_);

  return Status::OK();
}
}  // namespace onnxruntime