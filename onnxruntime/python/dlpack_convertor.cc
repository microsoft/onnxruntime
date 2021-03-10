// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ENABLE_TRAINING
#include "python/dlpack_convertor.h"

namespace onnxruntime {
namespace python {

DLDataType get_dlpack_data_type(const OrtValue& ort_value) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only tensor-type OrtValues are supported");
  DLDataType dtype;
  dtype.lanes = 1;
  const Tensor& tensor = ort_value.Get<Tensor>();
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

DLContext get_dlpack_context(const OrtValue& ort_value, const int64_t& device_id) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only OrtValues that are Tensors are currently supported");
  DLContext ctx;
  ctx.device_id = static_cast<int>(device_id);
  const Tensor& tensor = ort_value.Get<Tensor>();
  const auto& location = tensor.Location();
  switch (location.device.Type()) {
    case OrtDevice::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case OrtDevice::GPU:
#ifdef USE_ROCM
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLGPU;
#endif
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
DLManagedTensor* ort_value_to_dlpack(const OrtValue& ort_value) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only tensor type OrtValues are supported");
  OrtDLManagedTensor* ort_dlmanaged_tensor(new OrtDLManagedTensor);
  const Tensor& tensor = ort_value.Get<Tensor>();
  ort_dlmanaged_tensor->handle = ort_value;
  ort_dlmanaged_tensor->tensor.manager_ctx = ort_dlmanaged_tensor;
  ort_dlmanaged_tensor->tensor.deleter = &deleter;
  ort_dlmanaged_tensor->tensor.dl_tensor.data = const_cast<void*>(tensor.DataRaw());
  ort_dlmanaged_tensor->tensor.dl_tensor.ctx = get_dlpack_context(ort_value, tensor.Location().device.Id());
  ort_dlmanaged_tensor->tensor.dl_tensor.ndim = static_cast<int>(tensor.Shape().NumDimensions());
  ort_dlmanaged_tensor->tensor.dl_tensor.dtype = get_dlpack_data_type(ort_value);
  ort_dlmanaged_tensor->tensor.dl_tensor.shape =
      tensor.Shape().NumDimensions() > 0 ? const_cast<int64_t*>(&tensor.Shape()[0]) : nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.strides = nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.byte_offset = 0;
  return &(ort_dlmanaged_tensor->tensor);
}

}  // namespace python
}  // namespace onnxruntime
#endif