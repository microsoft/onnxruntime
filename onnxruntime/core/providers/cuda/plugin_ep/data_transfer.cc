
// Copyright (c) Microsoft Corporation. All rights reserved.

#include "core/providers/cuda/plugin_ep/data_transfer.h"
#include "core/providers/cuda/plugin_ep/utils.h"

namespace cuda_plugin_ep {
CudaDataTransferImpl::CudaDataTransferImpl() {
  ort_version_supported = ORT_API_VERSION;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
  Release = ReleaseImpl;
}

/*static*/
bool CudaDataTransferImpl::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const CudaDataTransferImpl*>(this_ptr);

  // logic copied from GPUDataTransfer::CanCopy
  OrtMemoryInfoDeviceType src_type = Shared::ep_api->MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_type = Shared::ep_api->MemoryDevice_GetDeviceType(dst_memory_device);
  auto src_vendor_id = Shared::ep_api->MemoryDevice_GetVendorId(src_memory_device);
  auto dst_vendor_id = Shared::ep_api->MemoryDevice_GetVendorId(dst_memory_device);

  if ((src_type == OrtMemoryInfoDeviceType_GPU && src_vendor_id != NvidiaVendorId) ||
      (dst_type == OrtMemoryInfoDeviceType_GPU && dst_vendor_id != NvidiaVendorId)) {
    return false;
  }

  // copy must be GPU to GPU or between GPU and CPU
  return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
         (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
         (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
}

/*static*/
OrtStatus* CudaDataTransferImpl::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors,
                                                 OrtValue** dst_tensors,
                                                 OrtSyncStream** streams,
                                                 size_t num_tensors) noexcept {
  auto& impl = *static_cast<CudaDataTransferImpl*>(this_ptr);
  bool need_stream_sync = false;

  for (size_t idx = 0; idx < num_tensors; ++idx) {
    const OrtValue* src_tensor = src_tensors[idx];
    OrtValue* dst_tensor = dst_tensors[idx];
    OrtSyncStream* stream = streams ? streams[idx] : nullptr;

    const OrtMemoryDevice* src_device = Shared::ep_api->Value_GetMemoryDevice(src_tensor);
    const OrtMemoryDevice* dst_device = Shared::ep_api->Value_GetMemoryDevice(dst_tensor);

    size_t bytes;
    RETURN_IF_ERROR(Shared::ort_api->GetTensorSizeInBytes(src_tensor, &bytes));

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    RETURN_IF_ERROR(Shared::ort_api->GetTensorData(src_tensor, &src_data));
    RETURN_IF_ERROR(Shared::ort_api->GetTensorMutableData(dst_tensor, &dst_data));

    OrtMemoryInfoDeviceType src_type = Shared::ep_api->MemoryDevice_GetDeviceType(src_device);
    OrtMemoryInfoDeviceType dst_type = Shared::ep_api->MemoryDevice_GetDeviceType(dst_device);
    OrtDeviceMemoryType src_mem_type = Shared::ep_api->MemoryDevice_GetMemoryType(src_device);
    OrtDeviceMemoryType dst_mem_type = Shared::ep_api->MemoryDevice_GetMemoryType(dst_device);

    const bool src_is_gpu_default = src_type == OrtMemoryInfoDeviceType_GPU &&
                                    src_mem_type == OrtDeviceMemoryType_DEFAULT;
    const bool dst_is_gpu_default = dst_type == OrtMemoryInfoDeviceType_GPU &&
                                    dst_mem_type == OrtDeviceMemoryType_DEFAULT;

    cudaStream_t cuda_stream = nullptr;
    if (stream) {
      cuda_stream = static_cast<cudaStream_t>(Shared::ort_api->SyncStream_GetHandle(stream));
    }

    if (dst_is_gpu_default) {
      if (src_is_gpu_default) {
        // Copy only if the two addresses are different.
        if (dst_data != src_data) {
          if (cuda_stream) {
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, cuda_stream));

          } else {
            CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));

            // For device memory to device memory copy, no host-side synchronization is performed by cudaMemcpy.
            // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
            need_stream_sync = true;
          }
        }
      } else {
        // copy from pinned or non-pinned CPU memory to GPU
        if (cuda_stream) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, cuda_stream));
        } else {
          CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));

          if (src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
            // For cudaMemcpy from pageable host memory to device memory, DMA to final destination may not
            // have completed.
            // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
            need_stream_sync = true;
          }
        }
      }
    } else if (src_is_gpu_default) {
      // copying from GPU to CPU memory, this is blocking

      if (cuda_stream) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, cuda_stream));

      } else {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
      }
    } else {
      // copying between CPU accessible memory

      if (dst_data != src_data) {
        if (cuda_stream) {
          if (src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
            // sync the stream first to make sure the data arrived
            CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
          }
        }

        memcpy(dst_data, src_data, bytes);
      }
    }
  }

  if (need_stream_sync) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
  }

  return nullptr;
}

/*static*/ void CudaDataTransferImpl::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {
  // no-op as we have a single shared instance in OrtEpFactory which is returned from CreateDataTransferImpl, and is
  // owned by and freed by the factory.
}

}  // namespace cuda_plugin_ep
