// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
#include <atomic>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_provider_factory.h"
#include "migraphx_execution_provider.h"
#include "migraphx_provider_factory_creator.h"
#include "migraphx_allocator.h"
#include "gpu_data_transfer.h"
#include "core/framework/provider_options.h"

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

struct ProviderInfo_MIGraphX_Impl final : ProviderInfo_MIGraphX {
  std::unique_ptr<IAllocator> CreateMIGraphXAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<MIGraphXAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<MIGraphXPinnedAllocator>(device_id, name);
  }

} g_info;

struct MIGraphX_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(device_id);
    info.target_device = "gpu";
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(options.device_id);
    info.target_device = "gpu";
    info.fp16_enable = options.migraphx_fp16_enable;
    info.fp8_enable = options.migraphx_fp8_enable;
    info.exhaustive_tune = options.migraphx_exhaustive_tune;
    info.int8_enable = options.migraphx_int8_enable;
    info.int8_calibration_table_name = "";
    if (options.migraphx_int8_calibration_table_name != nullptr) {
      info.int8_calibration_table_name = options.migraphx_int8_calibration_table_name;
    }
    info.int8_use_native_calibration_table = options.migraphx_use_native_calibration_table != 0;
    info.save_compiled_model = options.migraphx_save_compiled_model;
    info.save_model_file = "";
    if (options.migraphx_save_model_path != nullptr) {
      info.save_model_file = options.migraphx_save_model_path;
    }
    info.load_compiled_model = options.migraphx_load_compiled_model;
    info.load_model_file = "";
    if (options.migraphx_load_model_path != nullptr) {
      info.load_model_file = options.migraphx_load_model_path;
    }
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::MIGraphXExecutionProviderInfo::FromProviderOptions(options);
    auto& migx_options = *reinterpret_cast<OrtMIGraphXProviderOptions*>(provider_options);
    migx_options.device_id = internal_options.device_id;
    migx_options.migraphx_fp16_enable = internal_options.fp16_enable;
    migx_options.migraphx_fp8_enable = internal_options.fp8_enable;
    migx_options.migraphx_int8_enable = internal_options.int8_enable;
    migx_options.migraphx_exhaustive_tune = internal_options.exhaustive_tune;

    char* dest = nullptr;
    auto str_size = internal_options.int8_calibration_table_name.size();
    if (str_size == 0) {
      migx_options.migraphx_int8_calibration_table_name = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.int8_calibration_table_name.c_str(), str_size);
#else
      strncpy(dest, internal_options.int8_calibration_table_name.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      migx_options.migraphx_int8_calibration_table_name = (const char*)dest;
    }

    migx_options.migraphx_use_native_calibration_table = internal_options.int8_use_native_calibration_table;

    migx_options.migraphx_save_compiled_model = internal_options.save_compiled_model;
    migx_options.migraphx_save_model_path = internal_options.save_model_file.c_str();
    migx_options.migraphx_load_compiled_model = internal_options.load_compiled_model;
    migx_options.migraphx_load_model_path = internal_options.load_model_file.c_str();
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
