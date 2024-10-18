// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

#include "core/providers/webgpu/webgpu_provider_options.h"
using namespace onnxruntime::webgpu::options;

namespace onnxruntime {

struct WebGpuProviderFactory : IExecutionProviderFactory {
  WebGpuProviderFactory(int context_id, webgpu::WebGpuContext& context, WebGpuExecutionProviderInfo&& webgpu_ep_info)
      : context_id_{context_id}, context_{context}, info_{std::move(webgpu_ep_info)} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(context_id_, context_, std::move(info_));
  }

 private:
  int context_id_;
  webgpu::WebGpuContext& context_;
  WebGpuExecutionProviderInfo info_;
};

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(const ConfigOptions& config_options) {
  //
  // STEP.1 - prepare WebGpuExecutionProviderInfo
  //
  WebGpuExecutionProviderInfo webgpu_ep_info{
      // preferred layout is NHWC by default
      DataLayout::NHWC,
      // graph capture feature is disabled by default
      false,
  };

  std::string preferred_layout_str;
  if (config_options.TryGetConfigEntry(kPreferredLayout, preferred_layout_str)) {
    if (preferred_layout_str == kPreferredLayout_NHWC) {
      webgpu_ep_info.data_layout = DataLayout::NHWC;
    } else if (preferred_layout_str == kPreferredLayout_NCHW) {
      webgpu_ep_info.data_layout = DataLayout::NCHW;
    } else {
      ORT_THROW("Invalid preferred layout: ", preferred_layout_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP preferred layout: " << int(webgpu_ep_info.data_layout) << " (parsed from \""
                        << preferred_layout_str << "\")";

  std::string enable_graph_capture_str;
  if (config_options.TryGetConfigEntry(kEnableGraphCapture, enable_graph_capture_str)) {
    if (enable_graph_capture_str == kEnableGraphCapture_ON) {
      webgpu_ep_info.enable_graph_capture = true;
    } else if (enable_graph_capture_str == kEnableGraphCapture_OFF) {
      webgpu_ep_info.enable_graph_capture = false;
    } else {
      ORT_THROW("Invalid enable graph capture: ", enable_graph_capture_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP graph capture enable: " << webgpu_ep_info.enable_graph_capture;

  auto parse_buffer_cache_mode = [&config_options](const std::string& config_entry_str,
                                                   webgpu::BufferCacheMode default_value) -> webgpu::BufferCacheMode {
    std::string buffer_cache_mode_str;
    if (config_options.TryGetConfigEntry(config_entry_str, buffer_cache_mode_str)) {
      if (buffer_cache_mode_str == kBufferCacheMode_Disabled) {
        return webgpu::BufferCacheMode::Disabled;
      } else if (buffer_cache_mode_str == kBufferCacheMode_LazyRelease) {
        return webgpu::BufferCacheMode::LazyRelease;
      } else if (buffer_cache_mode_str == kBufferCacheMode_Simple) {
        return webgpu::BufferCacheMode::Simple;
      } else if (buffer_cache_mode_str == kBufferCacheMode_Bucket) {
        return webgpu::BufferCacheMode::Bucket;
      } else {
        ORT_THROW("Invalid buffer cache mode: ", config_entry_str);
      }
    } else {
      return default_value;
    }
  };

  webgpu_ep_info.storage_buffer_cache_mode = parse_buffer_cache_mode(kStorageBufferCacheMode, webgpu::BufferCacheMode::Bucket);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP storage buffer cache mode: " << webgpu_ep_info.storage_buffer_cache_mode;

  webgpu_ep_info.uniform_buffer_cache_mode = parse_buffer_cache_mode(kUniformBufferCacheMode, webgpu::BufferCacheMode::Simple);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP uniform buffer cache mode: " << webgpu_ep_info.uniform_buffer_cache_mode;

  webgpu_ep_info.query_resolve_buffer_cache_mode = parse_buffer_cache_mode(kQueryResolveBufferCacheMode, webgpu::BufferCacheMode::Disabled);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP query resolve buffer cache mode: " << webgpu_ep_info.query_resolve_buffer_cache_mode;

  webgpu_ep_info.default_buffer_cache_mode = parse_buffer_cache_mode(kDefaultBufferCacheMode, webgpu::BufferCacheMode::Disabled);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP default buffer cache mode: " << webgpu_ep_info.default_buffer_cache_mode;

  webgpu::ValidationMode validation_mode =
#ifndef NDEBUG
      webgpu::ValidationMode::Full  // for debug build, enable full validation by default
#else
      webgpu::ValidationMode::WGPUOnly  // for release build, only enable WGPU validation.
#endif  // !NDEBUG
      ;
  std::string validation_mode_str;
  if (config_options.TryGetConfigEntry(kValidationMode, validation_mode_str)) {
    if (validation_mode_str == kValidationMode_Disabled) {
      validation_mode = webgpu::ValidationMode::Disabled;
    } else if (validation_mode_str == kValidationMode_wgpuOnly) {
      validation_mode = webgpu::ValidationMode::WGPUOnly;
    } else if (validation_mode_str == kValidationMode_basic) {
      validation_mode = webgpu::ValidationMode::Basic;
    } else if (validation_mode_str == kValidationMode_full) {
      validation_mode = webgpu::ValidationMode::Full;
    } else {
      ORT_THROW("Invalid validation mode: ", validation_mode_str);
    }
  }

  // parse force CPU node names
  // The force CPU node names are separated by EOL (\n or \r\n) in the config entry.
  // each line is a node name that will be forced to run on CPU.
  std::string force_cpu_node_names_str;
  if (config_options.TryGetConfigEntry(kForceCpuNodeNames, force_cpu_node_names_str)) {
    std::vector<std::string> force_cpu_node_names;

    // split the string by EOL (\n or \r\n)
    std::istringstream ss(force_cpu_node_names_str);
    std::string line;
    while (std::getline(ss, line)) {
      // skip empty lines
      if (line.empty()) {
        continue;
      }

      force_cpu_node_names.push_back(line);
    }

    webgpu_ep_info.force_cpu_node_names = std::move(force_cpu_node_names);
  }

  //
  // STEP.2 - prepare WebGpuContext
  //
  int context_id = 0;
  std::string context_id_str;
  if (config_options.TryGetConfigEntry(kDeviceId, context_id_str)) {
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(context_id_str.data(), context_id_str.data() + context_id_str.size(), context_id).ec);
  }

  size_t webgpu_instance = 0;
  std::string webgpu_instance_str;
  if (config_options.TryGetConfigEntry(kWebGpuInstance, webgpu_instance_str)) {
    static_assert(sizeof(WGPUInstance) == sizeof(size_t), "WGPUInstance size mismatch");
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(webgpu_instance_str.data(), webgpu_instance_str.data() + webgpu_instance_str.size(), webgpu_instance).ec);
  }

  size_t webgpu_adapter = 0;
  std::string webgpu_adapter_str;
  if (config_options.TryGetConfigEntry(kWebGpuAdapter, webgpu_adapter_str)) {
    static_assert(sizeof(WGPUAdapter) == sizeof(size_t), "WGPUAdapter size mismatch");
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(webgpu_adapter_str.data(), webgpu_adapter_str.data() + webgpu_adapter_str.size(), webgpu_adapter).ec);
  }

  size_t webgpu_device = 0;
  std::string webgpu_device_str;
  if (config_options.TryGetConfigEntry(kWebGpuDevice, webgpu_device_str)) {
    static_assert(sizeof(WGPUDevice) == sizeof(size_t), "WGPUDevice size mismatch");
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(webgpu_device_str.data(), webgpu_device_str.data() + webgpu_device_str.size(), webgpu_device).ec);
  }

  auto& context = webgpu::WebGpuContextFactory::CreateContext(context_id,
                                                              reinterpret_cast<WGPUInstance>(webgpu_instance),
                                                              reinterpret_cast<WGPUAdapter>(webgpu_adapter),
                                                              reinterpret_cast<WGPUDevice>(webgpu_device),
                                                              validation_mode);
  context.Initialize(webgpu_ep_info);

  return std::make_shared<WebGpuProviderFactory>(context_id, context, std::move(webgpu_ep_info));
}

}  // namespace onnxruntime
