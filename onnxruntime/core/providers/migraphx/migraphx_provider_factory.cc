// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_provider_factory.h"
#include "migraphx_execution_provider.h"
#include "hip_allocator.h"
#include "gpu_data_transfer.h"
#include "core/framework/provider_options.h"
#include <atomic>

#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;


namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct MIGraphXProviderFactory : IExecutionProviderFactory {
  MIGraphXProviderFactory(const MIGraphXExecutionProviderInfo& info) : info_{info} {}
  ~MIGraphXProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  MIGraphXExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> MIGraphXProviderFactory::CreateProvider() {
  return std::make_unique<MIGraphXExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(const MIGraphXExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::MIGraphXProviderFactory>(info);
}

struct MIGraphX_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = device_id;
    info.target_device = "gpu";
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    MIGraphXExecutionProviderInfo info;
    info.device_id = options.device_id;
    info.target_device = "gpu";
    info.fp16_enable = options.migraphx_fp16_enable;
    info.int8_enable = options.migraphx_int8_enable;
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::MIGraphXExecutionProviderInfo::FromProviderOptions(options);
    auto& trt_options = *reinterpret_cast<OrtMIGraphXProviderOptions*>(provider_options);
    trt_options.device_id = internal_options.device_id;
    trt_options.migraphx_fp16_enable = internal_options.fp16_enable;
    trt_options.migraphx_int8_enable = internal_options.int8_enable;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    return onnxruntime::MIGraphXExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}

}
