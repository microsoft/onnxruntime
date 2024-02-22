// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/generator/constant_of_shape_base.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0,
    ConstantOfShapeDefaultOutputTypes);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 20, Output, 0,
    ConstantOfShapeDefaultOutputTypesOpset20);

// pytorch converter uses ConstantOfShape with int64 to create Pad input
// https://github.com/pytorch/pytorch/blob/044b519a80459f6787f6723c1c091a18b153d184/torch/onnx/symbolic_opset11.py#L449
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0,
    int64_t);

}  // namespace op_kernel_type_control

namespace {

using EnabledOutputTypes =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0);

using EnabledOutputTypesOpset20 =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 20, Output, 0);

class ConstantOfShape final : public ConstantOfShapeBase<EnabledOutputTypes>, public OpKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};

template <class T>
inline void FilloutOutput(T value, void* output_data, size_t size) {
  std::fill_n(reinterpret_cast<T*>(output_data), size, value);
}

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {
  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));

  auto output_data = output_tensor->MutableDataRaw();
  const void* value_ptr = GetValuePtr();
  const auto size = output_tensor->Shape().Size();
  const auto element_size = output_tensor->DataType()->Size();
  switch (element_size) {
    case sizeof(int8_t):
      FilloutOutput(*(reinterpret_cast<const int8_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    case sizeof(int16_t):
      FilloutOutput(*(reinterpret_cast<const int16_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    case sizeof(int32_t):
      FilloutOutput(*(reinterpret_cast<const int32_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    case sizeof(int64_t):
      FilloutOutput(*(reinterpret_cast<const int64_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported output datatype with size: ", element_size);
  }

  return Status::OK();
}

}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConstantOfShape,
    9,
    19,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypes>()),
    ConstantOfShape);

ONNX_CPU_OPERATOR_KERNEL(
    ConstantOfShape,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypesOpset20>()),
    ConstantOfShape);
}  // namespace onnxruntime
