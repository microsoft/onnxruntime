// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "constant_of_shape.h"
#include "core/providers/common.h"
#include "gsl/span"

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
  const auto tensor_type = GetTensorType();
  switch (tensor_type) {
    case TensorProto::BOOL:
      cuda::Fill(reinterpret_cast<bool*>(output_data), GetAttrValue().GetFromUnsigned<bool>(), size);
      break;
    case TensorProto::FLOAT:
      cuda::Fill(reinterpret_cast<float*>(output_data), GetAttrValue().GetFloat(), size);
      break;
    case TensorProto::FLOAT16:
      cuda::Fill(reinterpret_cast<half*>(output_data), reinterpret_cast<half&>(GetAttrValue().GetFloat16()), size);
      break;
    case TensorProto::DOUBLE:
      cuda::Fill(reinterpret_cast<double*>(output_data), GetAttrValue().GetDouble(), size);
      break;
    case TensorProto::INT8:
      cuda::Fill(reinterpret_cast<int8_t*>(output_data), GetAttrValue().GetFromSigned<int8_t>(), size);
      break;
    case TensorProto::INT16:
      cuda::Fill(reinterpret_cast<int16_t*>(output_data), GetAttrValue().GetFromSigned<int16_t>(), size);
      break;
    case TensorProto::INT32:
      cuda::Fill(reinterpret_cast<int32_t*>(output_data), GetAttrValue().GetFromSigned<int32_t>(), size);
      break;
    case TensorProto::INT64:
      cuda::Fill(reinterpret_cast<int64_t*>(output_data), GetAttrValue().GetFromSigned<int64_t>(), size);
      break;
    case TensorProto::UINT8:
      cuda::Fill(reinterpret_cast<uint8_t*>(output_data), GetAttrValue().GetFromUnsigned<uint8_t>(), size);
      break;
    case TensorProto::UINT16:
      cuda::Fill(reinterpret_cast<uint16_t*>(output_data), GetAttrValue().GetFromUnsigned<uint16_t>(), size);
      break;
    case TensorProto::UINT32:
      cuda::Fill(reinterpret_cast<uint32_t*>(output_data), GetAttrValue().GetFromUnsigned<uint32_t>(), size);
      break;
    case TensorProto::UINT64:
      cuda::Fill(reinterpret_cast<uint64_t*>(output_data), GetAttrValue().GetFromUnsigned<uint64_t>(), size);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", GetTensorType());
      break;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
