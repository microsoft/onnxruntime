// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
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

struct TensorrtProviderFactory : IExecutionProviderFactory {
  TensorrtProviderFactory(const TensorrtExecutionProviderInfo& info) : info_{info} {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  void GetCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list);

 private:
  TensorrtExecutionProviderInfo info_;
};

void TensorrtProviderFactory::GetCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list) {
  custom_op_domain_list = info_.custom_op_domain_list;
}

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
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    TensorrtExecutionProviderInfo info;
    info.device_id = device_id;
    info.has_trt_options = false;

    common::Status status = CreateTensorRTCustomOpDomainList(info);
    if (!status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Failed to get TRT plugins from TRT plugin registration.";
    }
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

    common::Status status = CreateTensorRTCustomOpDomainList(info);
    if (!status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Failed to get TRT plugins from TRT plugin registration.";
    }

    return std::make_shared<TensorrtProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::TensorrtExecutionProviderInfo::FromProviderOptions(options);
    auto& trt_options = *reinterpret_cast<OrtTensorRTProviderOptionsV2*>(provider_options);
    trt_options.device_id = internal_options.device_id;
    trt_options.trt_max_partition_iterations = internal_options.max_partition_iterations;
    trt_options.trt_min_subgraph_size = internal_options.min_subgraph_size;
    trt_options.trt_max_workspace_size = internal_options.max_workspace_size;
    trt_options.trt_fp16_enable = internal_options.fp16_enable;
    trt_options.trt_int8_enable = internal_options.int8_enable;

    char* dest = nullptr;
    auto str_size = internal_options.int8_calibration_table_name.size();
    if (str_size == 0) {
      trt_options.trt_int8_calibration_table_name = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.int8_calibration_table_name.c_str(), str_size);
#else
      strncpy(dest, internal_options.int8_calibration_table_name.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_int8_calibration_table_name = (const char*)dest;
    }

    trt_options.trt_int8_use_native_calibration_table = internal_options.int8_use_native_calibration_table;
    trt_options.trt_dla_enable = internal_options.dla_enable;
    trt_options.trt_dla_core = internal_options.dla_core;
    trt_options.trt_dump_subgraphs = internal_options.dump_subgraphs;
    trt_options.trt_engine_cache_enable = internal_options.engine_cache_enable;

    str_size = internal_options.engine_cache_path.size();
    if (str_size == 0) {
      trt_options.trt_engine_cache_path = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.engine_cache_path.c_str(), str_size);
#else
      strncpy(dest, internal_options.engine_cache_path.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_engine_cache_path = (const char*)dest;
    }

    trt_options.trt_engine_decryption_enable = internal_options.engine_decryption_enable;

    str_size = internal_options.engine_decryption_lib_path.size();
    if (str_size == 0) {
      trt_options.trt_engine_decryption_lib_path = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.engine_decryption_lib_path.c_str(), str_size);
#else
      strncpy(dest, internal_options.engine_decryption_lib_path.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_engine_decryption_lib_path = (const char*)dest;
    }

    trt_options.trt_force_sequential_engine_build = internal_options.force_sequential_engine_build;
    trt_options.trt_context_memory_sharing_enable = internal_options.context_memory_sharing_enable;
    trt_options.trt_layer_norm_fp32_fallback = internal_options.layer_norm_fp32_fallback;
    trt_options.trt_timing_cache_enable = internal_options.timing_cache_enable;
    trt_options.trt_force_timing_cache = internal_options.force_timing_cache;
    trt_options.trt_detailed_build_log = internal_options.detailed_build_log;
    trt_options.trt_build_heuristics_enable = internal_options.build_heuristics_enable;
    trt_options.trt_sparsity_enable = internal_options.sparsity_enable;
    trt_options.trt_builder_optimization_level = internal_options.builder_optimization_level;
    trt_options.trt_auxiliary_streams = internal_options.auxiliary_streams;
    str_size = internal_options.tactic_sources.size();
    if (str_size == 0) {
      trt_options.trt_tactic_sources = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.tactic_sources.c_str(), str_size);
#else
      strncpy(dest, internal_options.tactic_sources.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_tactic_sources = (const char*)dest;
    }

    str_size = internal_options.profile_min_shapes.size();
    if (str_size == 0) {
      trt_options.trt_profile_min_shapes = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.profile_min_shapes.c_str(), str_size);
#else
      strncpy(dest, internal_options.profile_min_shapes.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_profile_min_shapes = (const char*)dest;
    }

    str_size = internal_options.profile_max_shapes.size();
    if (str_size == 0) {
      trt_options.trt_profile_max_shapes = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.profile_max_shapes.c_str(), str_size);
#else
      strncpy(dest, internal_options.profile_max_shapes.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_profile_max_shapes = (const char*)dest;
    }

    str_size = internal_options.profile_opt_shapes.size();
    if (str_size == 0) {
      trt_options.trt_profile_opt_shapes = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.profile_opt_shapes.c_str(), str_size);
#else
      strncpy(dest, internal_options.profile_opt_shapes.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      trt_options.trt_profile_opt_shapes = (const char*)dest;
    }

    trt_options.trt_cuda_graph_enable = internal_options.cuda_graph_enable;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtTensorRTProviderOptionsV2*>(provider_options);
    return onnxruntime::TensorrtExecutionProviderInfo::ToProviderOptions(options);
  }

  void GetCustomOpDomainList(IExecutionProviderFactory* factory, std::vector<OrtCustomOpDomain*>& custom_op_domains_ptr) override {
    TensorrtProviderFactory* trt_factory = reinterpret_cast<TensorrtProviderFactory*>(factory);
    trt_factory->GetCustomOpDomainList(custom_op_domains_ptr);
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
