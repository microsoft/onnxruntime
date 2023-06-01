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
  int cannGetDeviceCount() override {
    uint32_t num_devices = 0;
    CANN_CALL_THROW(aclrtGetDeviceCount(&num_devices));
    return num_devices;
  }

  void cannMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    CANN_CALL_THROW(aclrtMemcpy(dst, count, src, count, ACL_MEMCPY_HOST_TO_DEVICE));
    CANN_CALL_THROW(aclrtSynchronizeStream(0));
  }

  void cannMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    CANN_CALL_THROW(aclrtMemcpy(dst, count, src, count, ACL_MEMCPY_DEVICE_TO_HOST));
  }

  void CANNExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options,
                                                      CANNExecutionProviderInfo& info) override {
    info = CANNExecutionProviderInfo::FromProviderOptions(options);
  }

  std::shared_ptr<IExecutionProviderFactory>
  CreateExecutionProviderFactory(const CANNExecutionProviderInfo& info) override {
    return std::make_shared<CANNProviderFactory>(info);
  }

  std::shared_ptr<IAllocator> CreateCannAllocator(int16_t device_id, size_t npu_mem_limit,
                                                  onnxruntime::ArenaExtendStrategy arena_extend_strategy,
                                                  OrtArenaCfg* default_memory_arena_cfg) override {
    return CANNExecutionProvider::CreateCannAllocator(device_id, npu_mem_limit, arena_extend_strategy,
                                                      default_memory_arena_cfg);
  }
} g_info;

struct CANN_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto params = reinterpret_cast<const OrtCANNProviderOptions*>(void_params);

    CANNExecutionProviderInfo info{};
    info.device_id = static_cast<OrtDevice::DeviceId>(params->device_id);
    info.npu_mem_limit = params->npu_mem_limit;
    info.arena_extend_strategy = params->arena_extend_strategy;
    info.enable_cann_graph = params->enable_cann_graph != 0;
    info.dump_graphs = params->dump_graphs != 0;
    info.precision_mode = params->precision_mode;
    info.op_select_impl_mode = params->op_select_impl_mode;
    info.optypelist_for_implmode = params->optypelist_for_implmode;
    info.default_memory_arena_cfg = params->default_memory_arena_cfg;

    return std::make_shared<CANNProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::CANNExecutionProviderInfo::FromProviderOptions(options);
    auto& cann_options = *reinterpret_cast<OrtCANNProviderOptions*>(provider_options);

    cann_options.device_id = internal_options.device_id;
    cann_options.npu_mem_limit = internal_options.npu_mem_limit;
    cann_options.arena_extend_strategy = internal_options.arena_extend_strategy;
    cann_options.enable_cann_graph = internal_options.enable_cann_graph;
    cann_options.dump_graphs = internal_options.dump_graphs;
    cann_options.precision_mode = internal_options.precision_mode;
    cann_options.op_select_impl_mode = internal_options.op_select_impl_mode;
    cann_options.optypelist_for_implmode = internal_options.optypelist_for_implmode;
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
