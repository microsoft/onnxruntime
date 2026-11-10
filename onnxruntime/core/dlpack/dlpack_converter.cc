// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/dlpack/dlpack_converter.h"

#include <algorithm>

#include "core/common/safeint.h"

namespace onnxruntime {
namespace dlpack {

namespace {

struct DataTypeMapping {
  ONNX_NAMESPACE::TensorProto_DataType ort_type;
  DLDataType dlpack_type;
};

// Sub-byte widths describe logical elements, not ORT's packed storage units.
constexpr DataTypeMapping data_type_mappings[] = {
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kDLFloat, 32, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT8, {kDLUInt, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_INT8, {kDLInt, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT16, {kDLUInt, 16, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_INT16, {kDLInt, 16, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_INT32, {kDLInt, 32, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_INT64, {kDLInt, 64, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_BOOL, {kDLBool, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {kDLFloat, 16, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, {kDLFloat, 64, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT32, {kDLUInt, 32, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT64, {kDLUInt, 64, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, {kDLBfloat, 16, 1}},
#if !defined(DISABLE_FLOAT8_TYPES)
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, {kDLFloat8_e4m3fn, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ, {kDLFloat8_e4m3fnuz, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, {kDLFloat8_e5m2, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ, {kDLFloat8_e5m2fnuz, 8, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0, {kDLFloat8_e8m0fnu, 8, 1}},
#endif
    {ONNX_NAMESPACE::TensorProto_DataType_UINT4, {kDLUInt, 4, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_INT4, {kDLInt, 4, 1}},
#if !defined(DISABLE_FLOAT4_TYPES)
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT4E2M1, {kDLFloat4_e2m1fn, 4, 1}},
#endif
    {ONNX_NAMESPACE::TensorProto_DataType_UINT2, {kDLUInt, 2, 1}},
    {ONNX_NAMESPACE::TensorProto_DataType_INT2, {kDLInt, 2, 1}},
};

DLDataType GetDlpackDataType(const OrtValue& ort_value) {
  ORT_ENFORCE(ort_value.IsTensor(), "Only tensor-type OrtValues are supported");
  const auto element_type = ort_value.Get<Tensor>().GetElementType();
  for (const auto& mapping : data_type_mappings) {
    if (mapping.ort_type == element_type) {
      return mapping.dlpack_type;
    }
  }
  ORT_THROW("Unsupported ORT data type for DLPack: ", element_type);
}

OrtDevice GetOrtDevice(const DLDevice& device) {
  switch (device.device_type) {
    case DLDeviceType::kDLCPU:
      return OrtDevice();
    case DLDeviceType::kDLCUDA:
      return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                       static_cast<OrtDevice::DeviceId>(device.device_id));
    case DLDeviceType::kDLROCM:
      return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD,
                       static_cast<OrtDevice::DeviceId>(device.device_id));
    default:
      ORT_THROW("Unsupported device type");
  }
}

MLDataType GetOrtValueDataType(const DLDataType& dtype) {
  if (dtype.lanes != 1) ORT_THROW("ORT does not support lanes != 1");
  for (const auto& mapping : data_type_mappings) {
    if (mapping.dlpack_type.code == dtype.code && mapping.dlpack_type.bits == dtype.bits) {
      return DataTypeImpl::TensorTypeFromONNXEnum(mapping.ort_type)->GetElementType();
    }
  }
  ORT_THROW("Unsupported DLPack data type: code ", static_cast<int>(dtype.code),
            ", bits ", static_cast<int>(dtype.bits));
}

const char* GetOrtDeviceName(const OrtDevice& device) {
  switch (device.Type()) {
    case OrtDevice::CPU:
      return CPU;
    case OrtDevice::GPU:
      return CUDA;
    default:
      ORT_THROW("Unknown device type: ", device.Type());
  }
}

bool IsContiguousTensor(const DLTensor& tensor) {
  if (!tensor.strides) {
    return true;
  }

  // Zero-size tensors (any dimension equals 0) have no elements, so any stride
  // layout is vacuously contiguous. Check upfront before validating strides,
  // because some frameworks (e.g. NumPy 2.x) set all strides to 0 for zero-size
  // tensors, which would otherwise fail the per-dimension stride check below.
  for (int i = 0; i < tensor.ndim; i++) {
    if (tensor.shape[i] == 0) {
      return true;
    }
  }

  int64_t running_size = 1;
  for (int i = tensor.ndim - 1; i >= 0; i--) {
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
      device.device_type = DLDeviceType::kDLCUDA;
      break;
    default:
      ORT_THROW("Cannot pack tensors on this device.");
  }

  return device;
}

struct OrtDLManagedTensor {
  OrtValue handle;
  DLManagedTensor tensor;
  TensorShapeVector strides;
};

static void DlpackDeleter(DLManagedTensor* arg) { delete static_cast<OrtDLManagedTensor*>(arg->manager_ctx); }

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
  auto ort_dlmanaged_tensor = std::make_unique<OrtDLManagedTensor>();

  Tensor& tensor = *ort_value.GetMutable<Tensor>();
#ifdef ENABLE_STRIDED_TENSORS
  ORT_ENFORCE(tensor.IsContiguous(), "DLPack export only supports contiguous tensors.");
#endif
  ort_dlmanaged_tensor->handle = ort_value;
  ort_dlmanaged_tensor->tensor.manager_ctx = ort_dlmanaged_tensor.get();
  ort_dlmanaged_tensor->tensor.deleter = &DlpackDeleter;
  ort_dlmanaged_tensor->tensor.dl_tensor.data = (tensor.MutableDataRaw());
  ort_dlmanaged_tensor->tensor.dl_tensor.device = GetDlpackDevice(ort_value, tensor.Location().device.Id());
  ort_dlmanaged_tensor->tensor.dl_tensor.ndim = static_cast<int>(tensor.Shape().NumDimensions());
  ort_dlmanaged_tensor->tensor.dl_tensor.dtype = GetDlpackDataType(ort_value);
  ort_dlmanaged_tensor->tensor.dl_tensor.shape =
      tensor.Shape().NumDimensions() > 0 ? &const_cast<TensorShape&>(tensor.Shape())[0] : nullptr;
  auto& strides = ort_dlmanaged_tensor->strides;
  strides.reserve(tensor.Shape().NumDimensions());
  // Empty tensors have no addressable elements, so zero strides avoid overflowing dimension products.
  const auto dims = tensor.Shape().GetDims();
  const bool is_empty = std::find(dims.begin(), dims.end(), 0) != dims.end();
  SafeInt<int64_t> stride = 1;
  for (size_t i = tensor.Shape().NumDimensions(); i > 0; --i) {
    strides.push_back(is_empty ? 0 : static_cast<int64_t>(stride));
    if (!is_empty) {
      stride *= tensor.Shape()[i - 1];
    }
  }
  std::reverse(strides.begin(), strides.end());
  ort_dlmanaged_tensor->tensor.dl_tensor.strides = strides.empty() ? nullptr : strides.data();
  ort_dlmanaged_tensor->tensor.dl_tensor.byte_offset = 0;
  auto* result = &ort_dlmanaged_tensor->tensor;
  ort_dlmanaged_tensor.release();
  return result;
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack) {
  ORT_ENFORCE(dlpack != nullptr, "DLPack tensor must not be null.");

  // ORT only supports contiguous tensor for now.
  ORT_ENFORCE(IsContiguousTensor(dlpack->dl_tensor), "ORT only supports contiguous tensor for now.");
  OrtDevice device = GetOrtDevice(dlpack->dl_tensor.device);
  MLDataType data_type = GetOrtValueDataType(dlpack->dl_tensor.dtype);
  OrtMemoryInfo info(GetOrtDeviceName(device), OrtDeviceAllocator, device);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      data_type, TensorShape(dlpack->dl_tensor.shape, static_cast<size_t>(dlpack->dl_tensor.ndim)),
      dlpack->dl_tensor.data, info, SafeInt<ptrdiff_t>(dlpack->dl_tensor.byte_offset));

  OrtValue ort_value;
  std::function<void(void*)> deleter = [dlpack](void* p) {
    if (dlpack->deleter != nullptr) {
      dlpack->deleter(dlpack);
    }
    auto deleter = DataTypeImpl::GetType<Tensor>()->GetDeleteFunc();
    if (deleter != NULL)
      deleter(p);
  };

  ort_value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), std::move(deleter));
  return ort_value;
}

}  // namespace dlpack
}  // namespace onnxruntime
