// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "nv_provider_factory.h"
#include <atomic>
#include "nv_execution_provider.h"
#include "nv_provider_factory_creator.h"
#include "core/framework/provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_custom_ops.h"
#include "nv_provider_options_internal.h"
#include <string.h>

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct ProviderInfo_Nv_Impl final : ProviderInfo_Nv {
  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  OrtStatus* UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) override {
    NvExecutionProviderInfo::UpdateProviderOptions(provider_options, options, string_copy);
    return nullptr;
  }

  OrtStatus* GetTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) override {
    common::Status status = CreateTensorRTCustomOpDomainList(domain_list, extra_plugin_lib_paths);
    if (!status.IsOK()) {
      return CreateStatus(ORT_FAIL, "[Nv EP] Can't create custom ops for TRT plugins.");
    }
    return nullptr;
  }

  OrtStatus* ReleaseCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list) override {
    ReleaseTensorRTCustomOpDomainList(domain_list);
    return nullptr;
  }

  OrtStatus* CreateProviderOptions(_Outptr_ OrtNvTensorRtRtxProviderOptions** out) override {
    if (!out) {
      return CreateStatus(ORT_INVALID_ARGUMENT, "Output pointer 'out' is NULL.");
    }
    *out = NULL;  // Initialize output

    struct OrtNvTensorRtRtxProviderOptions* options = (struct OrtNvTensorRtRtxProviderOptions*)malloc(sizeof(struct OrtNvTensorRtRtxProviderOptions));
    if (!options) {
      return CreateStatus(ORT_FAIL, "Failed to allocate memory for NvProviderOptions.");
    }
    options->magic_number = ORT_NV_PROVIDER_OPTIONS_MAGIC;  // Set the magic number

    *out = options;  // Assign the created handle to the output parameter
    return nullptr;
  }

  void ReleaseProviderOptions(_Frees_ptr_opt_ OrtNvTensorRtRtxProviderOptions* options) override {
    if (options) {
      if (options->magic_number == ORT_NV_PROVIDER_OPTIONS_MAGIC) {
        // If you had members that were heap-allocated *internally* by the options struct, free them here.
        // Example: free(options->some_string_option);

        // Invalidate the magic number to help detect use-after-free issues.
        options->magic_number = 0;

        // Free the main struct allocation
        free(options);
      } else {
        // Handle error: Log a warning? Abort in debug mode?
        // Attempting to release an invalid handle.
        // Depending on policy, you might just ignore it or log defensively.
      }
    }
  }

} g_info;

struct NvProviderFactory : IExecutionProviderFactory {
  NvProviderFactory(const NvExecutionProviderInfo& info) : info_{info} {}
  ~NvProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger);

 private:
  NvExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider() {
  return std::make_unique<NvExecutionProvider>(info_);
}

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider(const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
  const ConfigOptions& config_options = session_options.GetConfigOptions();
  const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

  // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
  // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
  // We extract those EP options to create a new "provider options" key/value map.
  std::string lowercase_ep_name = kNvTensorRTRTXExecutionProvider;
  std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  ProviderOptions provider_options;
  std::string key_prefix = "ep.";
  key_prefix += lowercase_ep_name;
  key_prefix += ".";

  for (const auto& [key, value] : config_options_map) {
    if (key.rfind(key_prefix, 0) == 0) {
      provider_options[key.substr(key_prefix.size())] = value;
    }
  }
  NvExecutionProviderInfo info = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(provider_options);

  auto ep = std::make_unique<NvExecutionProvider>(info);
  ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
  return ep;
}

struct Nv_Provider : Provider {
  void* GetInfo() override { return &g_info; }
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    NvExecutionProviderInfo info;
    info.device_id = device_id;
    info.has_trt_options = false;

    return std::make_shared<NvProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* /*provider_options*/) override {
    // auto& options = *reinterpret_cast<const OrtNvTensorRtRtxProviderOptions*>(provider_options);
    NvExecutionProviderInfo info;
    info.device_id = 0;
    info.has_user_compute_stream = false;
    info.user_compute_stream = 0;
    info.has_trt_options = true;
    info.max_partition_iterations = 0;
    info.min_subgraph_size = 0;
    info.max_workspace_size = 0;
    info.fp16_enable = false;
    info.int8_enable = false;
    info.int8_calibration_table_name = "";
    info.int8_use_native_calibration_table = false;
    info.dla_enable = false;
    info.dla_core = 0;
    info.dump_subgraphs = false;
    info.engine_cache_enable = false;
    info.engine_cache_path = "";
    info.weight_stripped_engine_enable = false;
    info.onnx_model_folder_path = "";
    info.engine_decryption_enable = false;
    info.engine_decryption_lib_path = "";
    info.force_sequential_engine_build = false;
    info.context_memory_sharing_enable = false;
    info.layer_norm_fp32_fallback = false;
    info.timing_cache_enable = false;
    info.timing_cache_path = "";
    info.force_timing_cache = false;
    info.detailed_build_log = false;
    info.build_heuristics_enable = false;
    info.sparsity_enable = 0;
    info.builder_optimization_level = 0;
    info.auxiliary_streams = -1;
    info.tactic_sources = "";
    info.extra_plugin_lib_paths = "";
    info.profile_min_shapes = "";
    info.profile_max_shapes = "";
    info.profile_opt_shapes = "";
    info.cuda_graph_enable = false;
    info.dump_ep_context_model = false;
    info.ep_context_file_path = "";
    info.ep_context_embed_mode = 0;
    info.engine_cache_prefix = "";
    info.engine_hw_compatible = false;
    info.onnx_bytestream = nullptr;
    info.onnx_bytestream_size = 0;
    info.op_types_to_exclude = "";

    return std::make_shared<NvProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    NvExecutionProviderInfo::UpdateProviderOptions(provider_options, options, true);
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtNvTensorRtRtxProviderOptions*>(provider_options);
    return onnxruntime::NvExecutionProviderInfo::ToProviderOptions(options);
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
