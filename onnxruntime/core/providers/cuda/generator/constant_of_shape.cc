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

ConstantOfShape::ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), OpKernel(info) {
  switch (tensor_type_) {
    case TensorProto::BOOL:
      constant_bool_ = cuda::CreateConstantValue<bool>(value_.GetFromUnsigned<bool>());
      break;
    case TensorProto::FLOAT:
      constant_float_ = cuda::CreateConstantValue<float>(value_.GetFloat());
      break;
    case TensorProto::FLOAT16:
      constant_half_ = cuda::CreateConstantValue<half>(reinterpret_cast<half&>(value_.GetFloat16()));
      break;
    case TensorProto::DOUBLE:
      constant_double_ = cuda::CreateConstantValue<double>(value_.GetDouble());
      break;
    case TensorProto::INT8:
      constant_int8_ = cuda::CreateConstantValue<int8_t>(value_.GetFromSigned<int8_t>());
      break;
    case TensorProto::INT16:
      constant_int16_ = cuda::CreateConstantValue<int16_t>(value_.GetFromSigned<int16_t>());
      break;
    case TensorProto::INT32:
      constant_int32_ = cuda::CreateConstantValue<int32_t>(value_.GetFromSigned<int32_t>());
      break;
    case TensorProto::INT64:
      constant_int64_ = cuda::CreateConstantValue<int64_t>(value_.GetFromSigned<int64_t>());
      break;
    case TensorProto::UINT8:
      constant_uint8_ = cuda::CreateConstantValue<uint8_t>(value_.GetFromUnsigned<uint8_t>());
      break;
    case TensorProto::UINT16:
      constant_uint16_ = cuda::CreateConstantValue<uint16_t>(value_.GetFromUnsigned<uint16_t>());
      break;
    case TensorProto::UINT32:
      constant_uint32_ = cuda::CreateConstantValue<uint32_t>(value_.GetFromUnsigned<uint32_t>());
      break;
    case TensorProto::UINT64:
      constant_uint64_ = cuda::CreateConstantValue<uint64_t>(value_.GetFromUnsigned<uint64_t>());
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", tensor_type_);
      break;
  }
}

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {
  auto shape_tensor = ctx->Input<Tensor>(0);

  if (shape_tensor->DataType() != DataTypeImpl::GetType<int64_t>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input tensor expected to contain int64 data");
  }

  auto& input_shape = shape_tensor->Shape();

  // If empty the output is a scalar with empty shape
  // TensorShape::Size() will still return 1 and we will output
  // one value
  std::vector<int64_t> output_dims;
  if (input_shape.NumDimensions() > 0) {
    auto span = gsl::make_span(shape_tensor->Data<int64_t>(), input_shape.Size());
    output_dims.insert(output_dims.end(), span.cbegin(), span.cend());
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  auto output_data = output_tensor->MutableDataRaw();
  auto size = output_shape.Size();

  switch (tensor_type_) {
    case TensorProto::BOOL:
      cudaMemcpyAsync(output_data, constant_bool_->GetBuffer(size), size * sizeof(bool), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::FLOAT:
      cudaMemcpyAsync(output_data, constant_float_->GetBuffer(size), size * sizeof(float), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::FLOAT16:
      cudaMemcpyAsync(output_data, constant_half_->GetBuffer(size), size * sizeof(half), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::DOUBLE:
      cudaMemcpyAsync(output_data, constant_double_->GetBuffer(size), size * sizeof(double), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::INT8:
      cudaMemcpyAsync(output_data, constant_int8_->GetBuffer(size), size * sizeof(int8_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::INT16:
      cudaMemcpyAsync(output_data, constant_int16_->GetBuffer(size), size * sizeof(int16_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::INT32:
      cudaMemcpyAsync(output_data, constant_int32_->GetBuffer(size), size * sizeof(int32_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::INT64:
      cudaMemcpyAsync(output_data, constant_int64_->GetBuffer(size), size * sizeof(int64_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::UINT8:
      cudaMemcpyAsync(output_data, constant_uint8_->GetBuffer(size), size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::UINT16:
      cudaMemcpyAsync(output_data, constant_uint16_->GetBuffer(size), size * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::UINT32:
      cudaMemcpyAsync(output_data, constant_uint32_->GetBuffer(size), size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
      break;
    case TensorProto::UINT64:
      cudaMemcpyAsync(output_data, constant_uint64_->GetBuffer(size), size * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
      break;
    default:
      ORT_THROW("Unsupported value attribute datatype: ", tensor_type_);
      break;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
