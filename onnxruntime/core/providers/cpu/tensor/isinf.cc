// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"

#include <cmath>

namespace onnxruntime {
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsInf

class IsInf final : public OpKernel {
 public:
  explicit IsInf(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t detect_positive_{1};
  int64_t detect_negative_{1};
};

ONNX_CPU_OPERATOR_KERNEL(
    IsInf,
    10,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    IsInf);

IsInf::IsInf(const OpKernelInfo& info) : OpKernel(info) {
  Status status = info.GetAttr("detect_positive", &detect_positive_);
  ORT_ENFORCE(status.IsOK(), "Failed to obtain detect_positive");
  status = info.GetAttr("detect_negative", &detect_negative_);
  ORT_ENFORCE(status.IsOK(), "Failed to obtain detect_negative");
}

namespace isinf_internal {
template <class T>
void ComputeImpl(const Tensor& X, Tensor& Y, bool detect_positive, bool detect_negative) {
  const auto total_items = X.Shape().Size();
  auto output_data = Y.template MutableData<bool>();

  if (detect_positive && detect_negative) {
    EigenMap<bool>(Y) = EigenMap<T>(X).array().isInf();
  } else if (detect_positive) {
    auto input_data = X.template Data<T>();
    auto end_data = input_data + total_items;
    std::transform(
        input_data, end_data, output_data, [](T v) {
          return (v == std::numeric_limits<T>::infinity());
        });

  } else if (detect_negative) {
    auto input_data = X.template Data<T>();
    auto end_data = input_data + total_items;
    std::transform(
        input_data, end_data, output_data, [](T v) {
          return (v == -std::numeric_limits<T>::infinity());
        });
  } else {
    // all false
    memset(output_data, false, total_items);
  }
}
}  // namespace isinf_internal

Status IsInf::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);
  const auto& X = *X_ptr;
  const auto& shape = X.Shape();
  auto& Y = *context->Output(0, shape);

  using namespace isinf_internal;

  auto dtype = X.DataType();
  if (utils::IsPrimitiveDataType<float>(dtype)) {
    ComputeImpl<float>(X, Y, detect_positive_ != 0, detect_negative_ != 0);
  } else if (utils::IsPrimitiveDataType<double>(dtype)) {
    ComputeImpl<double>(X, Y, detect_positive_ != 0, detect_negative_ != 0);
  } else {
    // should not reach this as no kernel is registered for this condition to be triggered - just an additional safety check
    ORT_THROW("Data type X must be float or double, but instead got ", dtype);
  }

  return Status::OK();
}

}  // namespace onnxruntime
