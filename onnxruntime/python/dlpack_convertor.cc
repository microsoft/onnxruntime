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

OrtDevice get_ort_device(const DLContext& ctx) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      return OrtDevice();
    case DLDeviceType::kDLGPU:
      return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(ctx.device_id));
    default:
      ORT_THROW("Unsupported device type");
  }
}

MLDataType get_ort_value_data_type(const DLDataType& dtype) {
  if (dtype.lanes != 1) ORT_THROW("ORT does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          return DataTypeImpl::GetType<bool>();
        default:
          ORT_THROW("Unsupported kUInt bits " + std::to_string(dtype.bits));
      }
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          return DataTypeImpl::GetType<int8_t>();
        case 16:
          return DataTypeImpl::GetType<int16_t>();
        case 32:
          return DataTypeImpl::GetType<int32_t>();
        case 64:
          return DataTypeImpl::GetType<int64_t>();
        default:
          ORT_THROW("Unsupported kInt bits " + std::to_string(dtype.bits));
      }
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          return DataTypeImpl::GetType<MLFloat16>();
        case 32:
          return DataTypeImpl::GetType<float>();
        case 64:
          return DataTypeImpl::GetType<double>();
        default:
          ORT_THROW("Unsupported kFloat bits " + std::to_string(dtype.bits));
      }
    default:
      ORT_THROW("Unsupported code " + std::to_string(dtype.code));
  }
}

static const char* get_device_name(const OrtDevice& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return CPU;
    case OrtDevice::GPU:
      return CUDA;
    default:
      ORT_THROW("Unknown device type: ", device.Type());
  }
}

static bool is_contiguous_tensor(const DLTensor& tensor) {
  if (!tensor.strides) {
    return true;
  }

  int64_t running_size = 1;
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    if (tensor.strides[i] != running_size) {
      return false;
    }

    running_size *= tensor.shape[i];
  }

  return true;
}

OrtValue dlpack_to_ort_value(const DLManagedTensor* src) {
  // ORT only supports contiguous tensor for now.
  ORT_ENFORCE(is_contiguous_tensor(src->dl_tensor), "ORT only supports contiguous tensor for now.");
  OrtDevice device = get_ort_device(src->dl_tensor.ctx);
  MLDataType data_type = get_ort_value_data_type(src->dl_tensor.dtype);
  auto deleter = [src](void*) { src->deleter(const_cast<DLManagedTensor*>(src)); };
  OrtMemoryInfo info(get_device_name(device), OrtDeviceAllocator, device, device.Id());
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(
      data_type, TensorShape(src->dl_tensor.shape, static_cast<size_t>(src->dl_tensor.ndim)), src->dl_tensor.data,
      info);

  OrtValue ort_value;
  ort_value.Initialize(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), deleter);
  return ort_value;
}

}  // namespace python
}  // namespace onnxruntime
#endif