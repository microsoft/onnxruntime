// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/dl_convertor.h"

namespace onnxruntime {
namespace python {

DLDataType get_dlpack_data_type(const OrtValue& ml_value) {
  ORT_ENFORCE(ml_value.IsTensor(), "Only OrtValues that are Tensors are currently supported");
  DLDataType dtype;
  dtype.lanes = 1;
  const Tensor& tensor = ml_value.Get<Tensor>();
  switch (tensor.GetElementType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      dtype.code = DLDataTypeCode::kDLFloat;
      dtype.bits = sizeof(double);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      dtype.code = DLDataTypeCode::kDLFloat;
      dtype.bits = sizeof(float);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      dtype.code = DLDataTypeCode::kDLInt;
      dtype.bits = sizeof(int8_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      dtype.code = DLDataTypeCode::kDLInt;
      dtype.bits = sizeof(int16_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      dtype.code = DLDataTypeCode::kDLInt;
      dtype.bits = sizeof(int);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      dtype.code = DLDataTypeCode::kDLInt;
      dtype.bits = sizeof(int64_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      dtype.code = DLDataTypeCode::kDLFloat;
      dtype.bits = sizeof(MLFloat16);
      break;
    // Currently bool is same as uint8 on both code and bits.
    // PyTorch's to_dlpack also does this, but in from_dlpack,
    // a torch.uint8 tensor is generated. This limitation from
    // PyTorch means we cannot create a torch.bool tensor
    // from a DLPack tensor.
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      dtype.code = DLDataTypeCode::kDLUInt;
      dtype.bits = sizeof(bool);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      dtype.code = DLDataTypeCode::kDLUInt;
      dtype.bits = sizeof(uint8_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      dtype.code = DLDataTypeCode::kDLUInt;
      dtype.bits = sizeof(uint16_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      dtype.code = DLDataTypeCode::kDLUInt;
      dtype.bits = sizeof(uint32_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      dtype.code = DLDataTypeCode::kDLUInt;
      dtype.bits = sizeof(uint64_t);
      break;
    default:
      ORT_THROW("Unexpected data type of ", tensor.GetElementType());
  }

  dtype.bits *= 8;  // bits.
  return dtype;
}

DLContext get_dlpack_context(const OrtValue& ml_value, const int64_t& device_id) {
  ORT_ENFORCE(ml_value.IsTensor(), "Only OrtValues that are Tensors are currently supported");
  DLContext ctx;
  ctx.device_id = device_id;
  const Tensor& tensor = ml_value.Get<Tensor>();
  const auto& location = tensor.Location();
  switch (location.device.Type()) {
    case OrtDevice::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case OrtDevice::GPU:
      ctx.device_type = DLDeviceType::kDLGPU;
      break;
    default:
      ORT_THROW("Cannot pack tensors on this device.");
  }

  return ctx;
}

struct OrtDLManagedTensor {
  OrtValue handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor* arg) { delete static_cast<OrtDLManagedTensor*>(arg->manager_ctx); }

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of OrtValue.
DLManagedTensor* ort_value_to_dlpack(const OrtValue& ml_value) {
  ORT_ENFORCE(ml_value.IsTensor(), "Only OrtValues that are Tensors are currently supported");
  OrtDLManagedTensor* ort_dlmanaged_tensor(new OrtDLManagedTensor);
  const Tensor& tensor = ml_value.Get<Tensor>();
  ort_dlmanaged_tensor->handle = ml_value;
  ort_dlmanaged_tensor->tensor.manager_ctx = ort_dlmanaged_tensor;
  ort_dlmanaged_tensor->tensor.deleter = &deleter;
  ort_dlmanaged_tensor->tensor.dl_tensor.data = const_cast<void*>(tensor.DataRaw());
  ort_dlmanaged_tensor->tensor.dl_tensor.ctx = get_dlpack_context(ml_value, tensor.Location().device.Id());
  ort_dlmanaged_tensor->tensor.dl_tensor.ndim = tensor.Shape().NumDimensions();
  ort_dlmanaged_tensor->tensor.dl_tensor.dtype = get_dlpack_data_type(ml_value);
  ort_dlmanaged_tensor->tensor.dl_tensor.shape =
      tensor.Shape().NumDimensions() > 0 ? const_cast<int64_t*>(&tensor.Shape()[0]) : nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.strides = nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.byte_offset = 0;
  return &(ort_dlmanaged_tensor->tensor);
}

}  // namespace python
}  // namespace onnxruntime
