// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/migraphx/migraphx_provider_factory.h"
#include <atomic>
#include "migraphx_execution_provider.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"

using namespace onnxruntime;

namespace onnxruntime {
struct MIGraphXProviderFactory : IExecutionProviderFactory {
  MIGraphXProviderFactory(int device_id) : device_id_(device_id) {}
  ~MIGraphXProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = device_id_;
    info.target_device = "gpu";
    return std::make_unique<MIGraphXExecutionProvider>(info);
  }

private:
  int device_id_;
};

// std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(int device_id) {
//   return std::make_shared<onnxruntime::MIGraphXProviderFactory>(device_id);
// }

}  // namespace onnxruntime

//ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, int device_id) {
//  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_MIGraphX(device_id));
//  return nullptr;
//}

ORT_API_STATUS_IMPL(OrtApis::CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
                  enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out) {
(void)name1;
(void)type;
(void)id1;
(void)mem_type1;
(*out) = nullptr;
//if (strcmp(name1, onnxruntime::CPU) == 0) {
//  *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), id1, mem_type1);
//} else if (strcmp(name1, onnxruntime::CUDA) == 0) {
//  *out = new OrtMemoryInfo(
//      onnxruntime::CUDA, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)), id1,
//      mem_type1);
//} else if (strcmp(name1, onnxruntime::CUDA_PINNED) == 0) {
//  *out = new OrtMemoryInfo(
//      onnxruntime::CUDA_PINNED, type, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, static_cast<OrtDevice::DeviceId>(id1)),
//      id1, mem_type1);
//} else {
//  return nullptr;
// //  return CreateStatus(ORT_INVALID_ARGUMENT, "Specified device is not supported.");
//}
return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type,
                   _Outptr_ OrtMemoryInfo** out) {
 *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), 0, mem_type);
 return nullptr;
}

ORT_API(void, OrtApis::ReleaseMemoryInfo, _Frees_ptr_opt_ OrtMemoryInfo* p) { delete p; }

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out) {
  *out = ptr->name;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out) {
  *out = ptr->id;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out) {
  *out = ptr->mem_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out) {
  *out = ptr->alloc_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2,
                    _Out_ int* out) {
  *out = (*info1 == *info2) ? 0 : -1;
  return nullptr;
}
