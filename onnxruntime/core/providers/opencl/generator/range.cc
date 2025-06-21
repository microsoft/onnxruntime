// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "range.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME range_kernel_src
#include "opencl_generated/generator/kernels/range.cl.inc"
}  // namespace

class Range : public OpenCLKernel {
 public:
  explicit Range(const OpKernelInfo& info) : OpenCLKernel(info) {
    LoadProgram(range_kernel_src, range_kernel_src_len);
    LoadKernel("Range_float");
    LoadKernel("Range_int");
    LoadKernel("Range_long");
    LoadKernel("Range_double");
  }

  Status Compute(OpKernelContext* context) const override;

  template <typename T>
  Status ComputeImp_(OpKernelContext* context) const;
};

template <>
Status Range::ComputeImp_<float>(OpKernelContext* context) const {
  const auto& start_tensor = context->RequiredInput<Tensor>(0);
  const auto& limit_tensor = context->RequiredInput<Tensor>(1);
  const auto* delta_tensor_ptr = context->Input<Tensor>(2);

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
  float start = *start_tensor.Data<float>();
  float limit = *limit_tensor.Data<float>();
  float delta = (delta_tensor_ptr == nullptr) ? float{1} : *(delta_tensor_ptr->Data<float>());

  if (delta == float{0}) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }
  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0)
    n = 0;
  TensorShape shape = {n};
  auto* Y = context->Output(0, shape);
  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Range_float")}
          .SetBuffer(*Y)
          .SetArg<float>(start)
          .SetArg<float>(delta)
          .SetArg<cl_long>(n)
          .Launch(*exec_, {n, 1, 1}));

  return Status::OK();
}

template <>
Status Range::ComputeImp_<int64_t>(OpKernelContext* context) const {
  const auto& start_tensor = context->RequiredInput<Tensor>(0);
  const auto& limit_tensor = context->RequiredInput<Tensor>(1);
  const auto* delta_tensor_ptr = context->Input<Tensor>(2);

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
  int64_t start = *start_tensor.Data<int64_t>();
  int64_t limit = *limit_tensor.Data<int64_t>();
  int64_t delta = (delta_tensor_ptr == nullptr) ? int64_t{1} : *(delta_tensor_ptr->Data<int64_t>());

  if (delta == int64_t{0}) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }
  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0)
    n = 0;
  TensorShape shape = {n};
  auto* Y = context->Output(0, shape);

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Range_long")}
          .SetBuffer(*Y)
          .SetArg<cl_long>(start)
          .SetArg<cl_long>(delta)
          .SetArg<cl_long>(n)
          .Launch(*exec_, {n, 1, 1}));

  return Status::OK();
}

Status Range::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }

  auto element_type = input_tensor->GetElementType();

  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return ComputeImp_<float>(ctx);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return ComputeImp_<int64_t>(ctx);
    // case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    //   return ComputeImp_<double>(ctx);
    // case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    //   return ComputeImp_<int32_t>(ctx);
    // case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    //   return ComputeImp_<int16_t>(ctx);
    default:
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Unsupported input type for Range operator.");
  }
}

ONNX_OPENCL_OPERATOR_KERNEL(
    Range,
    11,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // start
        .InputMemoryType(OrtMemTypeCPUInput, 1)  // limit
        .InputMemoryType(OrtMemTypeCPUInput, 2)  // delta
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int16_t>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()}),
    Range);

}  // namespace opencl
}  // namespace onnxruntime
