// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/generator/range.h"

#include <cmath>

#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Range, Input, 0,
    float, double, int16_t, int32_t, int64_t);
}

using RangeDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Range, Input, 0);
using EnabledRangeDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Range, Input, 0);

// Register a kernel for kMsDomain (contrib op) Range
#ifndef DISABLE_CONTRIB_OPS

namespace contrib {
// TODO: Remove this contrib kernel registration and the schema from the appropriate places
// once Keras Mask RCNN is shipped with all ONNX domain ops

// Currently this kernel is required to support Keras Mask-RCNN
ONNX_OPERATOR_KERNEL_EX(
    Range,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<RangeDataTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledRangeDataTypes>()),
    Range);

}  // namespace contrib

#endif

ONNX_CPU_OPERATOR_KERNEL(
    Range,
    11,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<RangeDataTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledRangeDataTypes>()),
    Range);

template <typename T>
static Status ComputeRange(OpKernelContext* ctx) {
  const auto& start_tensor = *ctx->Input<Tensor>(0);
  const auto& limit_tensor = *ctx->Input<Tensor>(1);
  const auto* delta_tensor_ptr = ctx->Input<Tensor>(2);

  if (!start_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "start in Range operator should be scalar like tensor, yet got shape:",
                           start_tensor.Shape());
  }
  if (!limit_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "limit in Range operator should be scalar like tensor, yet got shape:",
                           limit_tensor.Shape());
  }
  if (delta_tensor_ptr != nullptr && !delta_tensor_ptr->Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "delta in Range operator should be scalar like tensor, yet got shape:",
                           delta_tensor_ptr->Shape());
  }

  T start = *start_tensor.template Data<T>();
  T limit = *limit_tensor.template Data<T>();
  T delta = (delta_tensor_ptr == nullptr) ? T{1} : *(delta_tensor_ptr->template Data<T>());

  if (delta == T{0}) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }
  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0)
    n = 0;
  TensorShape shape = {n};
  T* y = ctx->Output(0, shape)->template MutableData<T>();
  for (int64_t i = 0; i < n; ++i) {
    *y++ = start;
    start += delta;
  }

  return Status::OK();
}

namespace range_internal {
template <class T>
struct CallRangeImpl {
  Status operator()(OpKernelContext* ctx) const {
    return ComputeRange<T>(ctx);
  }
};
}  // namespace range_internal

Status Range::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  utils::MLTypeCallDispatcherFromTypeList<EnabledRangeDataTypes> t_disp(input_tensor->GetElementType());
  return t_disp.InvokeRet<Status, range_internal::CallRangeImpl>(ctx);
}

}  // namespace onnxruntime
