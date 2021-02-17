// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/generator/constant_of_shape_base.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0,
    ConstantOfShapeDefaultOutputTypes);
}

namespace {

using SupportedOutputTypes =
    ORT_OP_KERNEL_ARG_SUPPORTED_TYPE_LIST_ALL_OPSETS(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0);

using EnabledOutputTypes =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0);

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
      FilloutOutput(*(reinterpret_cast<const int8_t*>(value_ptr)), output_data, size);
      break;
    case sizeof(int16_t):
      FilloutOutput(*(reinterpret_cast<const int16_t*>(value_ptr)), output_data, size);
      break;
    case sizeof(int32_t):
      FilloutOutput(*(reinterpret_cast<const int32_t*>(value_ptr)), output_data, size);
      break;
    case sizeof(int64_t):
      FilloutOutput(*(reinterpret_cast<const int64_t*>(value_ptr)), output_data, size);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported output datatype with size: ", element_size);
  }

  return Status::OK();
}

}  // namespace

ONNX_CPU_OPERATOR_KERNEL(
    ConstantOfShape,
    9,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<SupportedOutputTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypes>()),
    ConstantOfShape);

}  // namespace onnxruntime
