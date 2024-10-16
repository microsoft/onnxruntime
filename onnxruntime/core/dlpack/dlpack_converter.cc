// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/dlpack/dlpack_converter.h"

namespace onnxruntime {
namespace dlpack {

namespace {

DLDataType GetDlpackDataType(const OrtValue& ort_value) {
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
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      dtype.bits = sizeof(BFloat16);
      break;
    default:
      ORT_THROW("Unexpected data type of ", tensor.GetElementType());
  }

  dtype.bits *= 8;  // bits.
  return dtype;
}

OrtDevice GetOrtDevice(const DLDevice& device) {
  switch (device.device_type) {
    case DLDeviceType::kDLCPU:
      return OrtDevice();
    case DLDeviceType::kDLCUDA:
    case DLDeviceType::kDLROCM:
      return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(device.device_id));
    case DLDeviceType::kDLMAIA:
      return OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(device.device_id));
    default:
      ORT_THROW("Unsupported device type");
  }
}

MLDataType GetOrtValueDataType(const DLDataType& dtype, bool is_bool_tensor) {
  if (dtype.lanes != 1) ORT_THROW("ORT does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          return is_bool_tensor ? DataTypeImpl::GetType<bool>() : DataTypeImpl::GetType<uint8_t>();
        case 16:
          return DataTypeImpl::GetType<uint16_t>();
        case 32:
          return DataTypeImpl::GetType<uint32_t>();
        case 64:
          return DataTypeImpl::GetType<uint64_t>();
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
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          return DataTypeImpl::GetType<BFloat16>();
        default:
          ORT_THROW("Unsupported kBFloat bits " + std::to_string(dtype.bits));
      }
    default:
      ORT_THROW("Unsupported code " + std::to_string(dtype.code));
  }
}

const char* GetOrtDeviceName(const OrtDevice& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return CPU;
    case OrtDevice::GPU:
      return CUDA;
    case OrtDevice::FPGA:
      return "fpga";
    case OrtDevice::NPU:
      return "npu";
    default:
      ORT_THROW("Unknown device type: ", device.Type());
  }
}

bool IsContiguousTensor(const DLTensor& tensor) {
  if (!tensor.strides) {
    return true;
  }

  int64_t running_size = 1;
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    if (tensor.shape[i] == 0) {
      return true;
    }

    if (tensor.shape[i] != 1 && tensor.strides[i] != running_size) {
      return false;
    }

    running_size *= tensor.shape[i];
  }

  return true;
}

}  // namespace

DLDevice GetDlpackDevice(const OrtValue& ort_value, const int64_t& device_id) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only OrtValues that are Tensors are currently supported");
  DLDevice device;
  device.device_id = static_cast<int>(device_id);
  const Tensor& tensor = ort_value.Get<Tensor>();
  const auto& location = tensor.Location();
  switch (location.device.Type()) {
    case OrtDevice::CPU:
      device.device_type = DLDeviceType::kDLCPU;
      break;
    case OrtDevice::GPU:
#ifdef USE_ROCM
      device.device_type = DLDeviceType::kDLROCM;
#else
      device.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case OrtDevice::FPGA:
    case OrtDevice::NPU:
      device.device_type = DLDeviceType::kDLMAIA;
    default:
      ORT_THROW("Cannot pack tensors on this device.");
  }

  return device;
}

struct OrtDLManagedTensor {
  OrtValue handle;
  DLManagedTensor tensor;
};

static void DlpackDeleter(DLManagedTensor* arg) { delete static_cast<OrtDLManagedTensor*>(arg->manager_ctx); }

// This function should use smart pointers inside
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#pragma warning(disable : 26400)
#endif
// This function returns a pointer to DLManagedTensor constructed from an OrtValue
// The OrtValue inside OrtDLManagedTensor will increase its own buffer's ref count by one
// When the consumer of DLManagedTensor is done with the tensor, it should invoke the deleter.
DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only tensor type OrtValues are supported");
  OrtDLManagedTensor* ort_dlmanaged_tensor(new OrtDLManagedTensor);

  Tensor& tensor = *ort_value.GetMutable<Tensor>();
  ort_dlmanaged_tensor->handle = ort_value;
  ort_dlmanaged_tensor->tensor.manager_ctx = ort_dlmanaged_tensor;
  ort_dlmanaged_tensor->tensor.deleter = &DlpackDeleter;
  ort_dlmanaged_tensor->tensor.dl_tensor.data = (tensor.MutableDataRaw());
  ort_dlmanaged_tensor->tensor.dl_tensor.device = GetDlpackDevice(ort_value, tensor.Location().device.Id());
  ort_dlmanaged_tensor->tensor.dl_tensor.ndim = static_cast<int>(tensor.Shape().NumDimensions());
  ort_dlmanaged_tensor->tensor.dl_tensor.dtype = GetDlpackDataType(ort_value);
  ort_dlmanaged_tensor->tensor.dl_tensor.shape =
      tensor.Shape().NumDimensions() > 0 ? &const_cast<TensorShape&>(tensor.Shape())[0] : nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.strides = nullptr;
  ort_dlmanaged_tensor->tensor.dl_tensor.byte_offset = 0;
  return &(ort_dlmanaged_tensor->tensor);
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack, bool is_bool_tensor) {
  // ORT only supports contiguous tensor for now.
  ORT_ENFORCE(IsContiguousTensor(dlpack->dl_tensor), "ORT only supports contiguous tensor for now.");
  OrtDevice device = GetOrtDevice(dlpack->dl_tensor.device);
  MLDataType data_type = GetOrtValueDataType(dlpack->dl_tensor.dtype, is_bool_tensor);
  OrtMemoryInfo info(GetOrtDeviceName(device), OrtDeviceAllocator, device, device.Id());
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      data_type, TensorShape(dlpack->dl_tensor.shape, static_cast<size_t>(dlpack->dl_tensor.ndim)),
      dlpack->dl_tensor.data, info);

  OrtValue ort_value;
  std::function<void(void*)> deleter = [dlpack](void* p) {
    ORT_ENFORCE(dlpack->deleter != NULL, "A dlpack structure must have a deleter.");
    dlpack->deleter(dlpack);
    auto deleter = DataTypeImpl::GetType<Tensor>()->GetDeleteFunc();
    if (deleter != NULL)
      deleter(p);
  };

  ort_value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), deleter);
  return ort_value;
}

}  // namespace dlpack
}  // namespace onnxruntime
