// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "constant_of_shape.h"
#include "core/providers/common.h"
#include "gsl/gsl"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ConstantOfShape,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),
    ConstantOfShape);

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {
  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));
  auto output_data = output_tensor->MutableDataRaw();
  const auto size = output_tensor->Shape().Size();
  const void* value_ptr = GetValuePtr();
  const auto element_size = output_tensor->DataType()->Size();
  switch (element_size) {
    case sizeof(int8_t):
      cuda::Fill(reinterpret_cast<int8_t*>(output_data), *(reinterpret_cast<const int8_t*>(value_ptr)), size);
      break;
    case sizeof(int16_t):
      cuda::Fill(reinterpret_cast<int16_t*>(output_data), *(reinterpret_cast<const int16_t*>(value_ptr)), size);
      break;
    case sizeof(int32_t):
      cuda::Fill(reinterpret_cast<int32_t*>(output_data), *(reinterpret_cast<const int32_t*>(value_ptr)), size);
      break;
    case sizeof(int64_t):
      cuda::Fill(reinterpret_cast<int64_t*>(output_data), *(reinterpret_cast<const int64_t*>(value_ptr)), size);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype with sizeof=: ", element_size);
      break;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
