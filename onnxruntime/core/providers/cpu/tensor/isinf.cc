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
using IsInfTypesOpset10 = TypeList<float, double>;

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, IsInf, 10, Input, 0,
    IsInfTypesOpset10);

using IsInfTypesOpset20 =
    TypeList<
        float,
        double
#if !defined(DISABLE_FLOAT8_TYPES)
        ,
        Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ
#endif
        >;

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider,
    kOnnxDomain,
    IsInf,
    20,
    Input,
    0,
    IsInfTypesOpset20);
}  // namespace op_kernel_type_control

class IsInf final : public OpKernel {
 public:
  using EnabledDataTypes10 = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain,
                                                                 IsInf, 10, Input, 0);
  using EnabledDataTypes20 = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(kCpuExecutionProvider, kOnnxDomain,
                                                                 IsInf, 20, Input, 0);

  explicit IsInf(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t detect_positive_{1};
  int64_t detect_negative_{1};
  int opset_;
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    IsInf,
    10,
    19,
    KernelDefBuilder()
        .TypeConstraint("T1",
                        BuildKernelDefConstraintsFromTypeList<IsInf::EnabledDataTypes10>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    IsInf);

ONNX_CPU_OPERATOR_KERNEL(
    IsInf,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1",
                        BuildKernelDefConstraintsFromTypeList<IsInf::EnabledDataTypes20>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    IsInf);

IsInf::IsInf(const OpKernelInfo& info) : OpKernel(info) {
  Status status = info.GetAttr("detect_positive", &detect_positive_);
  ORT_ENFORCE(status.IsOK(), "Failed to obtain detect_positive");
  status = info.GetAttr("detect_negative", &detect_negative_);
  ORT_ENFORCE(status.IsOK(), "Failed to obtain detect_negative");
  opset_ = info.node().SinceVersion();
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

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
struct ComputeDispatchTarget<Float8E4M3FN> {
  void operator()(const Tensor&, Tensor& Y, bool, bool) const {
    EigenMap<bool>(Y).array() = false;
  }
};

template <>
struct ComputeDispatchTarget<Float8E4M3FNUZ> {
  void operator()(const Tensor&, Tensor& Y, bool, bool) const {
    EigenMap<bool>(Y).array() = false;
  }
};

template <>
struct ComputeDispatchTarget<Float8E5M2> {
  void operator()(const Tensor& X, Tensor& Y, bool detect_positive, bool detect_negative) const {
    auto& dims = X.Shape();
    auto input = ConstEigenVectorMap<uint8_t>(static_cast<const uint8_t*>(static_cast<const void*>(X.Data<Float8E5M2>())), onnxruntime::narrow<size_t>(dims.Size()));
    auto output = EigenMap<bool>(Y);

    // S.11111.00
    if (detect_positive && detect_negative) {
      output.array() = input.array() == 0b01111100 || input.array() == 0b11111100;
    } else if (detect_positive) {
      output.array() = input.array() == 0b01111100;
    } else if (detect_negative) {
      output.array() = input.array() == 0b11111100;
    } else {
      output.array() = false;
    }
  }
};

template <>
struct ComputeDispatchTarget<Float8E5M2FNUZ> {
  void operator()(const Tensor&, Tensor& Y, bool, bool) const {
    EigenMap<bool>(Y).array() = false;
  }
};
#endif
}  // namespace isinf_internal

Status IsInf::Compute(OpKernelContext* context) const {
  const auto* X_ptr = context->Input<Tensor>(0);
  const auto& X = *X_ptr;
  const auto& shape = X.Shape();
  auto& Y = *context->Output(0, shape);

  using namespace isinf_internal;

  if (opset_ < 20) {
    utils::MLTypeCallDispatcherFromTypeList<EnabledDataTypes10> dispatcher{X.GetElementType()};
    dispatcher.Invoke<ComputeDispatchTarget>(X, Y, detect_positive_ != 0, detect_negative_ != 0);
  } else {
    utils::MLTypeCallDispatcherFromTypeList<EnabledDataTypes20> dispatcher{X.GetElementType()};
    dispatcher.Invoke<ComputeDispatchTarget>(X, Y, detect_positive_ != 0, detect_negative_ != 0);
  }

  return Status::OK();
}

}  // namespace onnxruntime
