// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_provider_factory.h"
#include "core/providers/cann/cann_provider_factory_creator.h"
#include "core/providers/cann/cann_provider_options.h"
#include "core/providers/cann/cann_execution_provider_info.h"
#include "core/providers/cann/cann_execution_provider.h"
#include "core/providers/cann/cann_allocator.h"
#include "core/providers/cann/npu_data_transfer.h"
#include "core/providers/cann/cann_call.h"

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct CANNProviderFactory : IExecutionProviderFactory {
  explicit CANNProviderFactory(const CANNExecutionProviderInfo& info) : info_(info) {}
  ~CANNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  CANNExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> CANNProviderFactory::CreateProvider() {
  return std::make_unique<CANNExecutionProvider>(info_);
}

struct ProviderInfo_CANN_Impl : ProviderInfo_CANN {
  void CANNExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options,
                                                      CANNExecutionProviderInfo& info) override {
    info = CANNExecutionProviderInfo::FromProviderOptions(options);
  }

  std::shared_ptr<IExecutionProviderFactory>
  CreateExecutionProviderFactory(const CANNExecutionProviderInfo& info) override {
    return std::make_shared<CANNProviderFactory>(info);
  }
} g_info;

struct CANN_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto params = reinterpret_cast<const OrtCANNProviderOptions*>(void_params);

    CANNExecutionProviderInfo info{};
    info.device_id = static_cast<OrtDevice::DeviceId>(params->device_id);
    info.max_opqueue_num = params->max_opqueue_num;
    info.npu_mem_limit = params->npu_mem_limit;
    info.arena_extend_strategy = params->arena_extend_strategy;
    info.do_copy_in_default_stream = params->do_copy_in_default_stream != 0;
    info.default_memory_arena_cfg = params->default_memory_arena_cfg;

    return std::make_shared<CANNProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::CANNExecutionProviderInfo::FromProviderOptions(options);
    auto& cann_options = *reinterpret_cast<OrtCANNProviderOptions*>(provider_options);

    cann_options.device_id = internal_options.device_id;
    cann_options.max_opqueue_num = internal_options.max_opqueue_num;
    cann_options.npu_mem_limit = internal_options.npu_mem_limit;
    cann_options.arena_extend_strategy = internal_options.arena_extend_strategy;
    cann_options.do_copy_in_default_stream = internal_options.do_copy_in_default_stream;
    cann_options.default_memory_arena_cfg = internal_options.default_memory_arena_cfg;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtCANNProviderOptions*>(provider_options);
    return onnxruntime::CANNExecutionProviderInfo::ToProviderOptions(options);
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
