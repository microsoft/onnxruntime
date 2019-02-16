// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/shrink.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    Shrink,
    9,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int64_t>(),
                                            DataTypeImpl::GetTensorType<uint64_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<uint32_t>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<uint16_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<MLFloat16>(),
                                            DataTypeImpl::GetTensorType<BFloat16>()}),
    Shrink);

namespace shrink_internal {
template <class T>
inline T ShrinkImpl(const T& val, float bias, float lambd) {
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

void ShrinkMLFloat16(const Tensor* input, Tensor* output, float bias, float lambd) {
  auto span = gsl::make_span(input->Data<MLFloat16>(), input->Shape().Size());
  auto output_data = output->template MutableData<MLFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [bias, lambd](const MLFloat16& val) {
    float fl = math::halfToFloat(val.val);
    return MLFloat16(math::floatToHalf(ShrinkImpl<float>(fl, bias, lambd)));
  });
}

void ShrinkBFloat16(const Tensor* input, Tensor* output, float bias, float lambd) {
  auto span = gsl::make_span(input->Data<BFloat16>(), input->Shape().Size());
  auto output_data = output->template MutableData<BFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [bias, lambd](const BFloat16& val) {
    float fl = val.ToFloat();
    return BFloat16(ShrinkImpl<float>(fl, bias, lambd));
  });
}

}  // namespace shrink_internal

Status Shrink::Compute(OpKernelContext* p_op_kernel_context) const {
  using namespace shrink_internal;

  auto input = p_op_kernel_context->Input<Tensor>(0);
  auto output = p_op_kernel_context->Output(0, input->Shape());

  auto dtype = input->DataType();

  if (dtype == DataTypeImpl::GetType<float>()) {
    EigenMap<float>(*output) = EigenMap<float>(*input).unaryExpr([this](const float& val) { return ShrinkImpl<float>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<double>()) {
    EigenMap<double>(*output) = EigenMap<double>(*input).unaryExpr([this](const double& val) { return ShrinkImpl<double>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<int64_t>()) {
    EigenMap<int64_t>(*output) = EigenMap<int64_t>(*input).unaryExpr([this](const int64_t& val) { return ShrinkImpl<int64_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<uint64_t>()) {
    EigenMap<uint64_t>(*output) = EigenMap<uint64_t>(*input).unaryExpr([this](const uint64_t& val) { return ShrinkImpl<uint64_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<int32_t>()) {
    EigenMap<int32_t>(*output) = EigenMap<int32_t>(*input).unaryExpr([this](const int32_t& val) { return ShrinkImpl<int32_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<uint32_t>()) {
    EigenMap<uint32_t>(*output) = EigenMap<uint32_t>(*input).unaryExpr([this](const uint32_t& val) { return ShrinkImpl<uint32_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<int16_t>()) {
    EigenMap<int16_t>(*output) = EigenMap<int16_t>(*input).unaryExpr([this](const int16_t& val) { return ShrinkImpl<int16_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<uint16_t>()) {
    EigenMap<uint16_t>(*output) = EigenMap<uint16_t>(*input).unaryExpr([this](const uint16_t& val) { return ShrinkImpl<uint16_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<int8_t>()) {
    EigenMap<int8_t>(*output) = EigenMap<int8_t>(*input).unaryExpr([this](const int8_t& val) { return ShrinkImpl<int8_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<uint8_t>()) {
    EigenMap<uint8_t>(*output) = EigenMap<uint8_t>(*input).unaryExpr([this](const uint8_t& val) { return ShrinkImpl<uint8_t>(val, bias_, lambd_); });
  } else if (dtype == DataTypeImpl::GetType<MLFloat16>()) {
    ShrinkMLFloat16(input, output, bias_, lambd_);
  } else if (dtype == DataTypeImpl::GetType<BFloat16>()) {
    ShrinkBFloat16(input, output, bias_, lambd_);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input types for the Shrink operator are constrained to all numeric types only");
  }

  return Status::OK();
}
}  // namespace onnxruntime
