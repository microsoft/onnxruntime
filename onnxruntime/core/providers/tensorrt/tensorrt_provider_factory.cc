// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"
#include "tensorrt_provider_factory_creator.h"
#include "core/framework/provider_options.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/providers/tensorrt/tensorrt_execution_provider_custom_ops.h"
#include <string.h>

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct ProviderInfo_TensorRT_Impl final : ProviderInfo_TensorRT {
  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  OrtStatus* UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) override {
    TensorrtExecutionProviderInfo::UpdateProviderOptions(provider_options, options, string_copy);
    return nullptr;
  }

  OrtStatus* GetTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) override {
    common::Status status = CreateTensorRTCustomOpDomainList(domain_list, extra_plugin_lib_paths);
    if (!status.IsOK()) {
      return CreateStatus(ORT_FAIL, "[TensorRT EP] Can't create custom ops for TRT plugins.");
    }
    return nullptr;
  }

  OrtStatus* ReleaseCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list) override {
    ReleaseTensorRTCustomOpDomainList(domain_list);
    return nullptr;
  }

} g_info;

struct TensorrtProviderFactory : IExecutionProviderFactory {
  TensorrtProviderFactory(const TensorrtExecutionProviderInfo& info) : info_{info} {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  TensorrtExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  return std::make_unique<TensorrtExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> TensorrtProviderFactoryCreator::Create(int device_id) {
  TensorrtExecutionProviderInfo info;
  info.device_id = device_id;
  info.has_trt_options = false;
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(info);
}

struct Tensorrt_Provider : Provider {
  void* GetInfo() override { return &g_info; }
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    TensorrtExecutionProviderInfo info;
    info.device_id = device_id;
    info.has_trt_options = false;

    return std::make_shared<TensorrtProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtTensorRTProviderOptionsV2*>(provider_options);
    TensorrtExecutionProviderInfo info;
    info.device_id = options.device_id;
    info.has_user_compute_stream = options.has_user_compute_stream != 0;
    info.user_compute_stream = options.user_compute_stream;
    info.has_trt_options = true;
    info.max_partition_iterations = options.trt_max_partition_iterations;
    info.min_subgraph_size = options.trt_min_subgraph_size;
    info.max_workspace_size = options.trt_max_workspace_size;
    info.fp16_enable = options.trt_fp16_enable != 0;
    info.int8_enable = options.trt_int8_enable != 0;
    info.int8_calibration_table_name = options.trt_int8_calibration_table_name == nullptr ? "" : options.trt_int8_calibration_table_name;
    info.int8_use_native_calibration_table = options.trt_int8_use_native_calibration_table != 0;
    info.dla_enable = options.trt_dla_enable != 0;
    info.dla_core = options.trt_dla_core;
    info.dump_subgraphs = options.trt_dump_subgraphs != 0;
    info.engine_cache_enable = options.trt_engine_cache_enable != 0;
    info.engine_cache_path = options.trt_engine_cache_path == nullptr ? "" : options.trt_engine_cache_path;
    info.engine_decryption_enable = options.trt_engine_decryption_enable != 0;
    info.engine_decryption_lib_path = options.trt_engine_decryption_lib_path == nullptr ? "" : options.trt_engine_decryption_lib_path;
    info.force_sequential_engine_build = options.trt_force_sequential_engine_build != 0;
    info.context_memory_sharing_enable = options.trt_context_memory_sharing_enable != 0;
    info.layer_norm_fp32_fallback = options.trt_layer_norm_fp32_fallback != 0;
    info.timing_cache_enable = options.trt_timing_cache_enable != 0;
    info.force_timing_cache = options.trt_force_timing_cache != 0;
    info.detailed_build_log = options.trt_detailed_build_log != 0;
    info.build_heuristics_enable = options.trt_build_heuristics_enable != 0;
    info.sparsity_enable = options.trt_sparsity_enable;
    info.builder_optimization_level = options.trt_builder_optimization_level;
    info.auxiliary_streams = options.trt_auxiliary_streams;
    info.tactic_sources = options.trt_tactic_sources == nullptr ? "" : options.trt_tactic_sources;
    info.extra_plugin_lib_paths = options.trt_extra_plugin_lib_paths == nullptr ? "" : options.trt_extra_plugin_lib_paths;
    info.profile_min_shapes = options.trt_profile_min_shapes == nullptr ? "" : options.trt_profile_min_shapes;
    info.profile_max_shapes = options.trt_profile_max_shapes == nullptr ? "" : options.trt_profile_max_shapes;
    info.profile_opt_shapes = options.trt_profile_opt_shapes == nullptr ? "" : options.trt_profile_opt_shapes;
    info.cuda_graph_enable = options.trt_cuda_graph_enable != 0;

    return std::make_shared<TensorrtProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    TensorrtExecutionProviderInfo::UpdateProviderOptions(provider_options, options, true);
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtTensorRTProviderOptionsV2*>(provider_options);
    return onnxruntime::TensorrtExecutionProviderInfo::ToProviderOptions(options);
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
