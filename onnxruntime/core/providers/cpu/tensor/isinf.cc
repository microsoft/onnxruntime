// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/common/common.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/math.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, IsInf, Input, 0,
    float, double);
}  // namespace op_kernel_type_control

class IsInf final : public OpKernel {
 public:
  using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                          IsInf, Input, 0);

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
        .TypeConstraint("T1",
                        BuildKernelDefConstraintsFromTypeList<IsInf::EnabledDataTypes>())
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
struct ComputeDispatchTarget {
  void operator()(const Tensor& X, Tensor& Y, bool detect_positive, bool detect_negative) const {
    const auto total_items = X.Shape().Size();
    auto output_data = Y.MutableData<bool>();

    if (detect_positive && detect_negative) {
      EigenMap<bool>(Y) = EigenMap<T>(X).array().isInf();
    } else if (detect_positive) {
      auto input_data = X.Data<T>();
      auto end_data = input_data + total_items;
      std::transform(
          input_data, end_data, output_data, [](T v) {
            return (v == std::numeric_limits<T>::infinity());
          });

    } else if (detect_negative) {
      auto input_data = X.Data<T>();
      auto end_data = input_data + total_items;
      std::transform(
          input_data, end_data, output_data, [](T v) {
            return (v == -std::numeric_limits<T>::infinity());
          });
    } else {
      // all false
      memset(output_data, false, onnxruntime::narrow<size_t>(total_items));
    }
  }
};
}  // namespace isinf_internal

Status IsInf::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);
  const auto& X = *X_ptr;
  const auto& shape = X.Shape();
  auto& Y = *context->Output(0, shape);

  using namespace isinf_internal;

  utils::MLTypeCallDispatcherFromTypeList<EnabledDataTypes> dispatcher{X.GetElementType()};
  dispatcher.Invoke<ComputeDispatchTarget>(X, Y, detect_positive_ != 0, detect_negative_ != 0);

  return Status::OK();
}

}  // namespace onnxruntime
