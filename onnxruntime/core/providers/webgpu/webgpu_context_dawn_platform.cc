// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_context_dawn_platform.h"

#include <algorithm>
#include <memory>
#include <thread>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "dawn/native/DawnNative.h"
#include "dawn/platform/DawnPlatform.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace onnxruntime::webgpu {
namespace {

uint32_t GetDawnWorkerThreadCount() {
  return std::max(2u, std::thread::hardware_concurrency() / 2u);
}

class DawnPlatform final : public dawn::platform::Platform {
 public:
  std::unique_ptr<dawn::platform::WorkerTaskPool> CreateWorkerTaskPool() override {
    return dawn::platform::WorkerTaskPool::CreateDawnDefault(GetDawnWorkerThreadCount());
  }
};

DawnPlatform& GetDawnPlatform() {
  // Dawn retains this non-owning pointer; keep it alive through instance teardown.
  static DawnPlatform* platform = new DawnPlatform();
  return *platform;
}

}  // namespace

const DawnProcTable& GetBundledDawnProcs() {
  return dawn::native::GetProcs();
}

wgpu::Instance CreateBundledDawnInstance(wgpu::InstanceDescriptor instance_desc) {
  dawn::native::DawnInstanceDescriptor dawn_instance_desc{};
  dawn_instance_desc.platform = &GetDawnPlatform();
  instance_desc.nextInChain = &dawn_instance_desc;
  return wgpu::CreateInstance(&instance_desc);
}

InlinedVector<wgpu::Adapter> EnumerateBundledDawnAdapters(WGPUInstance instance,
                                                          const wgpu::RequestAdapterOptions& options) {
  dawn::native::Instance native_instance(reinterpret_cast<dawn::native::InstanceBase*>(instance));
  const auto adapters = native_instance.EnumerateAdapters(&options);
  InlinedVector<wgpu::Adapter> result;
  result.reserve(adapters.size());
  for (const auto& adapter : adapters) {
    result.emplace_back(adapter.Get());
  }
  return result;
}

}  // namespace onnxruntime::webgpu
