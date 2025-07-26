// Copyright (c) Microsoft Corporation. All rights reserved.

#include "core/providers/cuda/plugin_ep/factory.h"

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/plugin_ep/allocator.h"
#include "core/providers/cuda/plugin_ep/sync_stream.h"
#include "core/providers/cuda/plugin_ep/utils.h"

namespace cuda_plugin_ep {

CudaEpFactory::CudaEpFactory() {
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

/*static*/
const char* CudaEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto& factory = *static_cast<const CudaEpFactory*>(this_ptr);
  return factory.ep_name.c_str();
}

/*static*/
const char* CudaEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto& factory = *static_cast<const CudaEpFactory*>(this_ptr);
  return factory.vendor.c_str();
}

/*static*/
uint32_t CudaEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const CudaEpFactory*>(this_ptr);
  return factory->vendor_id;
}

/*static*/
const char* CudaEpFactory::GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return ORT_VERSION;
}

/*static*/
OrtStatus* CudaEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                  const OrtHardwareDevice* const* devices,
                                                  size_t num_devices,
                                                  OrtEpDevice** ep_devices,
                                                  size_t max_ep_devices,
                                                  size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);

  int num_cuda_devices = 0;
  cudaGetDeviceCount(&num_cuda_devices);
  RETURN_IF_ERROR(factory.CreateMemoryInfoForDevices(num_cuda_devices));

  /* in theory we can match on the LUID in the OrtHardwareDevice metadata, but that requires the CUDA Driver API
  std::vector<uint64_t> device_to_luid;
  device_to_luid.resize(num_cuda_devices);

  for (int i = 0; i < num_cuda_devices; ++i) {
    CUdevice device;
    cuDeviceGet(&device, i);

    char luid[8];
    unsigned int nodeMask;
    if (cuDeviceGetLuid(luid, &nodeMask, device) == CUDA_SUCCESS) {
      device_to_luid[i] = *reinterpret_cast<uint64_t*>(luid);
    }
  }
  */

  // should never happen unless there's an issue with duplicate devices in the OrtHardwareDevice list
  const auto max_devices = std::min(num_devices, max_ep_devices);

  int16_t device_id = 0;
  for (size_t i = 0; i < num_devices && num_ep_devices < max_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (Shared::ort_api->HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
        Shared::ort_api->HardwareDevice_VendorId(&device) == 0x10de) {
      /* ideally we'd match on LUID here
         for now we use an incrementing device id. could be a mismatch if you have multiple different CUDA GPUs.
         alternative is to limit to one device only.

      // find the device id. On Windows we have the LUID in the OrtHardwareDevice metadata.
      const OrtKeyValuePairs* metadata = Shared::ort_api->HardwareDevice_Metadata(&device);
      const char* luid_str = Shared::ort_api->GetKeyValue(metadata, "LUID");

      if (!luid_str && num_devices > 1) {
        // if there's no LUID we can't match device
        return Shared::ort_api->CreateStatus(ORT_EP_FAIL, "OrtHardwareDevice does not have LUID");
      }

      char* luid_end = nullptr;
      uint64_t luid = std::strtoull(luid_str, &luid_end, 10);
      for (; device_id < num_cuda_devices; ++device_id) {
        if (device_to_luid[device_id] == luid) {
          break;
        }
      }

      if (device_id == num_cuda_devices) {
        std::string msg("Could not match LUID to a CUDA device. LUID=");
        msg += luid_str;

        return Shared::ort_api->CreateStatus(ORT_EP_FAIL, msg.c_str());
      }
      */

      // create the EP options and add the device id
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      Shared::ort_api->CreateKeyValuePairs(&ep_options);
      Shared::ort_api->AddKeyValuePair(ep_options, "device_id", std::to_string(device_id).c_str());

      // create the OrtEpDevice
      OrtEpDevice* ep_device = nullptr;
      RETURN_IF_ERROR(Shared::ort_api->GetEpApi()->CreateEpDevice(&factory, &device, ep_metadata, ep_options,
                                                                  &ep_device));

      Shared::ort_api->ReleaseKeyValuePairs(ep_options);

      const OrtMemoryInfo* gpu_mem_info = factory.gpu_memory_infos[device_id].get();
      const OrtMemoryInfo* host_accessible_mem_info = factory.host_accessible_memory_infos[device_id].get();

      RETURN_IF_ERROR(Shared::ep_api->EpDevice_AddAllocatorInfo(ep_device, gpu_mem_info));
      RETURN_IF_ERROR(Shared::ep_api->EpDevice_AddAllocatorInfo(ep_device, host_accessible_mem_info));

      ep_devices[num_ep_devices++] = ep_device;

      ++device_id;
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* CudaEpFactory::CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                       _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                       _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                       _In_ size_t /*num_devices*/,
                                       _In_ const OrtSessionOptions* /*session_options*/,
                                       _In_ const OrtLogger* /*logger*/,
                                       _Out_ OrtEp** /*ep*/) noexcept {
  return Shared::ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "CUDA EP factory does not support this method.");
}

/*static*/
void CudaEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
  // no-op as we never create an EP here.
}

/*static*/
OrtStatus* CudaEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                              const OrtMemoryInfo* memory_info,
                                              const OrtKeyValuePairs* /*allocator_options*/,
                                              OrtAllocator** allocator) noexcept {
  // this function is free to return the same allocator instance for all calls and make ReleaseAllocator a no-op
  // e.g. allocator instance is in unique_ptr in the OrtEpFactory instance.
  // ORT will create a shared allocator in the environment and the user can choose to use it in an inference session.
  // Otherwise ORT will create an allocator when adding the EP to an inference session.
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);

  auto cuda_allocator = std::make_unique<CudaOrtAllocator>(memory_info);
  *allocator = cuda_allocator.release();

  return nullptr;
}

/*static*/
void CudaEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept {
  delete static_cast<CudaOrtAllocator*>(allocator);
}

/*static*/
OrtStatus* CudaEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  *data_transfer = &factory.data_transfer_impl;

  return nullptr;
}

/*static*/
bool CudaEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;
}

/*static*/
OrtStatus* CudaEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                        const OrtMemoryDevice* memory_device,
                                                        const OrtKeyValuePairs* /*stream_options*/,
                                                        OrtSyncStreamImpl** ort_stream) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  auto device_id = Shared::ep_api->MemoryDevice_GetDeviceId(memory_device);

  // the OrtEpFactory could have a cache of stream instances if it wants to avoid creating a new one on every
  // call. the CudaStreamSyncImpl::Release could return the instance to the cache.
  cudaStream_t stream = nullptr;
  CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
  CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Currently this API is only used for creating a stream that is used outside of a session, as we're using the
  // 'real' CUDA IExecutionProvider implementation for the EP. Due to that we need to connect it up to an internal
  // onnxruntime::Stream that has the correct settings for the session.
  // We do that externally by passing the cudaStream_t in via the "user_compute_stream" provider option.
  //
  // For use within an inference session in a completely plugin EP we need to implement
  // OrtEp::CreateSyncStreamForDevice so that the session's CPU allocator is available, as well as the
  // session options such as whether graph capture is enabled.

  std::unique_ptr<CudaSyncStreamImpl> sync_stream;
  CUDA_RETURN_IF_ERROR(CudaSyncStreamImpl::Create(std::move(stream), *memory_device, sync_stream));

  *ort_stream = sync_stream.release();

  return nullptr;
}

OrtStatus* CudaEpFactory::CreateMemoryInfoForDevices(int num_devices) {
  gpu_memory_infos.reserve(num_devices);
  host_accessible_memory_infos.reserve(num_devices);

  for (int device_id = 0; device_id < num_devices; ++device_id) {
    OrtMemoryInfo* mem_info = nullptr;
    RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("CUDA", OrtMemoryInfoDeviceType_GPU,
                                                /*vendor*/ OrtDevice::VendorIds::NVIDIA,
                                                /* device_id */ device_id,
                                                OrtDeviceMemoryType_DEFAULT,
                                                /*alignment*/ 0,
                                                OrtAllocatorType::OrtDeviceAllocator,
                                                &mem_info));

    gpu_memory_infos.emplace_back(MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo));

    // HOST_ACCESSIBLE memory should use the non-CPU device type
    mem_info = nullptr;
    RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("CUDA host accessible", OrtMemoryInfoDeviceType_GPU,
                                                /*vendor*/ OrtDevice::VendorIds::NVIDIA,
                                                /* device_id */ device_id,
                                                OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                /*alignment*/ 0,
                                                OrtAllocatorType::OrtDeviceAllocator,
                                                &mem_info));

    host_accessible_memory_infos.emplace_back(MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo));
  }

  return nullptr;
}

}  // namespace cuda_plugin_ep
