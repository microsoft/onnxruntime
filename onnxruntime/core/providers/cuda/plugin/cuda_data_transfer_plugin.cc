// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_data_transfer_plugin.h"

namespace onnxruntime {
namespace cuda_plugin {

CudaDataTransfer::CudaDataTransfer(const OrtApi& ort_api, const OrtEpApi& ep_api,
                                   const OrtMemoryDevice* gpu_device)
    : OrtDataTransferImpl{},
      ort_api_(ort_api),
      ep_api_(ep_api),
      gpu_device_(gpu_device) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
}

/*static*/ void ORT_API_CALL CudaDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
  delete static_cast<CudaDataTransfer*>(this_ptr);
}

/*static*/ bool ORT_API_CALL CudaDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_device,
    const OrtMemoryDevice* dst_device) noexcept {
  auto* dt = static_cast<const CudaDataTransfer*>(this_ptr);
  const OrtEpApi& ep_api = dt->ep_api_;
  auto src_type = ep_api.MemoryDevice_GetDeviceType(src_device);
  auto dst_type = ep_api.MemoryDevice_GetDeviceType(dst_device);

  bool src_is_cpu = (src_type == OrtMemoryInfoDeviceType_CPU);
  bool dst_is_cpu = (dst_type == OrtMemoryInfoDeviceType_CPU);
  bool src_is_gpu = (src_type == OrtMemoryInfoDeviceType_GPU);
  bool dst_is_gpu = (dst_type == OrtMemoryInfoDeviceType_GPU);

  // Support CPU→GPU, GPU→CPU, GPU→GPU
  return (src_is_cpu && dst_is_gpu) ||
         (src_is_gpu && dst_is_cpu) ||
         (src_is_gpu && dst_is_gpu);
}

/*static*/ OrtStatus* ORT_API_CALL CudaDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl* this_ptr,
    const OrtValue** src_tensors,
    OrtValue** dst_tensors,
    OrtSyncStream** streams,
    size_t count) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* dt = static_cast<CudaDataTransfer*>(this_ptr);

  for (size_t i = 0; i < count; ++i) {
    Ort::ConstValue src{src_tensors[i]};
    Ort::UnownedValue dst{dst_tensors[i]};

    auto src_type_shape = src.GetTensorTypeAndShapeInfo();
    size_t count_elems = src_type_shape.GetElementCount();

    // Get element size from data type
    ONNXTensorElementDataType elem_type = src_type_shape.GetElementType();
    size_t elem_size = 0;
    switch (elem_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        elem_size = sizeof(float);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        elem_size = sizeof(double);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        elem_size = sizeof(int32_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        elem_size = sizeof(int64_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        elem_size = 2;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        elem_size = 1;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        elem_size = 2;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        elem_size = 4;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        elem_size = 8;
        break;
      default:
        return dt->ort_api_.CreateStatus(ORT_EP_FAIL, "Unsupported tensor element type for copy");
    }

    size_t bytes = count_elems * elem_size;
    if (bytes == 0) continue;

    const void* src_data = src.GetTensorRawData();
    void* dst_data = dst.GetTensorMutableRawData();

    // Determine copy direction
    const OrtMemoryInfo* src_mem_info = src.GetTensorMemoryInfo();
    const OrtMemoryInfo* dst_mem_info = dst.GetTensorMemoryInfo();
    const OrtMemoryDevice* src_dev = dt->ep_api_.MemoryInfo_GetMemoryDevice(src_mem_info);
    const OrtMemoryDevice* dst_dev = dt->ep_api_.MemoryInfo_GetMemoryDevice(dst_mem_info);
    auto src_dev_type = dt->ep_api_.MemoryDevice_GetDeviceType(src_dev);
    auto dst_dev_type = dt->ep_api_.MemoryDevice_GetDeviceType(dst_dev);

    cudaMemcpyKind copy_kind;
    if (src_dev_type == OrtMemoryInfoDeviceType_CPU && dst_dev_type == OrtMemoryInfoDeviceType_GPU) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (src_dev_type == OrtMemoryInfoDeviceType_GPU && dst_dev_type == OrtMemoryInfoDeviceType_CPU) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else if (src_dev_type == OrtMemoryInfoDeviceType_GPU && dst_dev_type == OrtMemoryInfoDeviceType_GPU) {
      copy_kind = cudaMemcpyDeviceToDevice;
    } else {
      return dt->ort_api_.CreateStatus(ORT_EP_FAIL, "Unsupported copy direction");
    }

    // Use async copy if stream is provided
    if (streams != nullptr && streams[i] != nullptr) {
      cudaStream_t cuda_stream = static_cast<cudaStream_t>(
          Ort::GetApi().SyncStream_GetHandle(streams[i]));
      PL_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, copy_kind, cuda_stream));
    } else {
      PL_CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, copy_kind));
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
