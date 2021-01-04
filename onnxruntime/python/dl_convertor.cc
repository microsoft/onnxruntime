// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/dl_convertor.h"
// #include "onnx-ml.pb.h"
#include "onnxruntime_cxx_api.h"

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
      ORT_THROW("Cannot pack tensors on this device: " + location.device.Type());
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
DLManagedTensor* ortvalue_to_dlpack(const OrtValue& ml_value) {
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

onnxruntime::MLDataType get_ortvalue_data_type(const DLDataType& dtype) {
  onnxruntime::MLDataType ml_dtype;

  if (dtype.lanes != 1) {
    ORT_THROW("ONNX Runtime does not support lanes != 1");
  }

  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          ml_dtype = DataTypeImpl::GetTensorType<uint8_t>();
          break;
        default:
          ORT_THROW("Unsupported kUInt bits " + dtype.bits);
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          ml_dtype = DataTypeImpl::GetTensorType<int8_t>();
          break;
        case 16:
          ml_dtype = DataTypeImpl::GetTensorType<int16_t>();
          break;
        case 32:
          ml_dtype = DataTypeImpl::GetTensorType<int>();
          break;
        case 64:
          ml_dtype = DataTypeImpl::GetTensorType<int64_t>();
          break;
        default:
          ORT_THROW("Unsupported kInt bits " + dtype.bits);
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          ml_dtype = DataTypeImpl::GetTensorType<MLFloat16>();
          break;
        case 32:
          ml_dtype = DataTypeImpl::GetTensorType<float>();
          break;
        case 64:
          ml_dtype = DataTypeImpl::GetTensorType<double>();
          break;
        default:
          ORT_THROW("Unsupported kFloat bits " + dtype.bits);
      }
      break;
    default:
      ORT_THROW("Unsupported code " + dtype.code);
  }

  return ml_dtype;
}

onnx::TensorProto_DataType get_ortvalue_data_type2(const DLDataType& dtype) {
  onnx::TensorProto_DataType ml_dtype;

  if (dtype.lanes != 1) {
    ORT_THROW("ONNX Runtime does not support lanes != 1");
  }

  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          ml_dtype = data_types_internal::ToTensorDataType<uint8_t>();
          break;
        default:
          ORT_THROW("Unsupported kUInt bits " + dtype.bits);
      }
      break;
    // case DLDataTypeCode::kDLInt:
    //   switch (dtype.bits) {
    //     case 8:
    //       ml_dtype = ToTensorDataType<int8_t>();
    //       break;
    //     case 16:
    //       ml_dtype = DataTypeImpl::GetTensorType<int16_t>();
    //       break;
    //     case 32:
    //       ml_dtype = DataTypeImpl::GetTensorType<int>();
    //       break;
    //     case 64:
    //       ml_dtype = DataTypeImpl::GetTensorType<int64_t>();
    //       break;
    //     default:
    //       ORT_THROW("Unsupported kInt bits " + dtype.bits);
    //   }
    //   break;
    // case DLDataTypeCode::kDLFloat:
    //   switch (dtype.bits) {
    //     case 16:
    //       ml_dtype = DataTypeImpl::GetTensorType<MLFloat16>();
    //       break;
    //     case 32:
    //       ml_dtype = DataTypeImpl::GetTensorType<float>();
    //       break;
    //     case 64:
    //       ml_dtype = DataTypeImpl::GetTensorType<double>();
    //       break;
    //     default:
    //       ORT_THROW("Unsupported kFloat bits " + dtype.bits);
    //   }
    //   break;
    default:
      ORT_THROW("Unsupported code " + dtype.code);
  }

  return ml_dtype;
}

// static OrtDevice getORTDevice(const DLContext& ctx) {
//   switch (ctx.device_type) {
//     case DLDeviceType::kDLCPU:
//       return OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0);
//     case DLDeviceType::kDLGPU:
//       return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, ctx.device_id);
//     case DLDeviceType::kDLCPUPinned:
//       return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::CUDA_PINNED, ctx.device_id);
//     default:
//       ORT_THROW("Unsupported device_type: " + ctx.device_type);
//   }
// }

void deleter2(void* arg) { delete static_cast<OrtDLManagedTensor*>((static_cast<DLManagedTensor*>(arg)->manager_ctx)); }

OrtValue ortvalue_from_dlpack(const DLManagedTensor* src, AllocatorPtr alloc) {
  // OrtDevice device = getORTDevice(src->dl_tensor.ctx);
  auto dtype = get_ortvalue_data_type2(src->dl_tensor.dtype);
  auto deleter3 = [src](void*) -> void *{
    src->deleter(const_cast<DLManagedTensor*>(src));
    return nullptr;
  };
  (void)deleter3;

    // return at::from_blob(src->dl_tensor.data,
    //     IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
    //     deleter,
    //     at::device(device).dtype(stype));

  // OrtValue p_mlvalue = OrtValue(static_cast<void*>(src->dl_tensor.data),
  //                               static_cast<onnxruntime::MLDataType>(dtype),
  //                               static_cast<onnxruntime::DeleteFunc>(deleter2));

  // auto x = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(dtype->TypeAsProto()->tensor_type().elem_type());
  OrtValue p_mlvalue = Ort::Value::CreateTensor(alloc, src->dl_tensor.shape, src->dl_tensor.ndim, dtype);
  return p_mlvalue;
}

}  // namespace python
}  // namespace onnxruntime
