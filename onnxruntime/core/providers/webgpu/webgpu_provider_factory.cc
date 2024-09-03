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
  WebGpuProviderFactory(int context_id, const webgpu::WebGpuContext& context, const WebGpuExecutionProviderInfo& webgpu_ep_info)
      : context_id_{context_id}, context_{context}, info_{webgpu_ep_info} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(context_id_, context_, info_);
  }

 private:
  int context_id_;
  const webgpu::WebGpuContext& context_;
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
    if (enable_graph_capture_str == kkEnableGraphCapture_ON) {
      webgpu_ep_info.enable_graph_capture = true;
    } else if (enable_graph_capture_str == kkEnableGraphCapture_OFF) {
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

  webgpu_ep_info.uniform_buffer_cache_mode = parse_buffer_cache_mode(kUniformBufferCacheMode, webgpu::BufferCacheMode::LazyRelease);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP uniform buffer cache mode: " << webgpu_ep_info.uniform_buffer_cache_mode;

  webgpu_ep_info.query_resolve_buffer_cache_mode = parse_buffer_cache_mode(kQueryResolveBufferCacheMode, webgpu::BufferCacheMode::Disabled);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP query resolve buffer cache mode: " << webgpu_ep_info.query_resolve_buffer_cache_mode;

  webgpu_ep_info.default_buffer_cache_mode = parse_buffer_cache_mode(kDefaultBufferCacheMode, webgpu::BufferCacheMode::Disabled);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP default buffer cache mode: " << webgpu_ep_info.default_buffer_cache_mode;

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
                                                              reinterpret_cast<WGPUDevice>(webgpu_device));
  context.Initialize(webgpu_ep_info);

  return std::make_shared<WebGpuProviderFactory>(context_id, context, webgpu_ep_info);
}

}  // namespace onnxruntime
