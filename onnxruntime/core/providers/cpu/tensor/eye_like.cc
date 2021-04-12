// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/eye_like.h"

#include "core/common/common.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, EyeLike, Output, 0,
    float, double, uint64_t, int64_t, int32_t);
}

using EyeLikeDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, EyeLike, Output, 0);
using EnabledEyeLikeDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, EyeLike, Output, 0);

ONNX_CPU_OPERATOR_KERNEL(
    EyeLike,
    9,
    KernelDefBuilder()
        .TypeConstraint(
            "T1",
            BuildKernelDefConstraintsFromTypeList<EyeLikeDataTypes>(),
            BuildKernelDefConstraintsFromTypeList<EnabledEyeLikeDataTypes>())
        .TypeConstraint(
            "T2",
            BuildKernelDefConstraintsFromTypeList<EyeLikeDataTypes>(),
            BuildKernelDefConstraintsFromTypeList<EnabledEyeLikeDataTypes>()),
    EyeLike);

namespace {
template <typename T>
struct ComputeDispatchTarget {
  void operator()(const int64_t k, Tensor& output) {
    const auto& output_shape = output.Shape();
    auto output_mat = EigenMatrixMapRowMajor<T>(
        output.template MutableData<T>(),
        output_shape[0],
        output_shape[1]);

    output_mat.setZero();

    if ((k >= 0 && k >= output_shape[1]) || (k < 0 && std::abs(k) >= output_shape[0])) {
      return;
    }

    output_mat.diagonal(k).array() = static_cast<T>(1);
  }
};
}  // namespace

Status EyeLike::Compute(OpKernelContext* context) const {
  const auto& T1 = context->RequiredInput<Tensor>(0);

  const auto& input_shape = T1.Shape();
  if (input_shape.NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "EyeLike : Input tensor dimension is not 2");
  }

  // set output tensor shape same as input tensor
  auto& T2 = context->RequiredOutput(0, input_shape);

  const auto output_tensor_dtype =
      has_dtype_ ? static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype_) : T1.GetElementType();

  utils::MLTypeCallDispatcherFromTypeList<EnabledEyeLikeDataTypes> dispatcher{output_tensor_dtype};
  dispatcher.Invoke<ComputeDispatchTarget>(k_, T2);

  return Status::OK();
}

}  // namespace onnxruntime
