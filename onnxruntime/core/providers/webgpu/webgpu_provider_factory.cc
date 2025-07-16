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
  WebGpuProviderFactory(int context_id, webgpu::WebGpuContext& context, WebGpuExecutionProviderConfig&& webgpu_ep_config)
      : context_id_{context_id}, context_{context}, config_{std::move(webgpu_ep_config)} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(context_id_, context_, std::move(config_));
  }

 private:
  int context_id_;
  webgpu::WebGpuContext& context_;
  WebGpuExecutionProviderConfig config_;
};

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(const ConfigOptions& config_options) {
  //
  // STEP.1 - prepare WebGpuExecutionProviderConfig
  //
  WebGpuExecutionProviderConfig webgpu_ep_config{
      // preferred layout is NHWC by default
      DataLayout::NHWC,
      // graph capture feature is disabled by default
      false,
      // enable pix capture feature is diabled by default
      false,
  };

  std::string preferred_layout_str;
  if (config_options.TryGetConfigEntry(kPreferredLayout, preferred_layout_str)) {
    if (preferred_layout_str == kPreferredLayout_NHWC) {
      webgpu_ep_config.data_layout = DataLayout::NHWC;
    } else if (preferred_layout_str == kPreferredLayout_NCHW) {
      webgpu_ep_config.data_layout = DataLayout::NCHW;
    } else {
      ORT_THROW("Invalid preferred layout: ", preferred_layout_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP preferred layout: " << int(webgpu_ep_config.data_layout) << " (parsed from \""
                        << preferred_layout_str << "\")";

  std::string enable_graph_capture_str;
  if (config_options.TryGetConfigEntry(kEnableGraphCapture, enable_graph_capture_str)) {
    if (enable_graph_capture_str == kEnableGraphCapture_ON) {
      webgpu_ep_config.enable_graph_capture = true;
    } else if (enable_graph_capture_str == kEnableGraphCapture_OFF) {
      webgpu_ep_config.enable_graph_capture = false;
    } else {
      ORT_THROW("Invalid enable graph capture: ", enable_graph_capture_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP graph capture enable: " << webgpu_ep_config.enable_graph_capture;

  // parse force CPU node names
  // The force CPU node names are separated by EOL (\n or \r\n) in the config entry.
  // each line is a node name that will be forced to run on CPU.
  std::string force_cpu_node_names_str;
  if (config_options.TryGetConfigEntry(kForceCpuNodeNames, force_cpu_node_names_str)) {
    // split the string by EOL (\n or \r\n)
    std::istringstream ss(force_cpu_node_names_str);
    std::string line;
    while (std::getline(ss, line)) {
      // skip empty lines
      if (line.empty()) {
        continue;
      }

      webgpu_ep_config.force_cpu_node_names.push_back(line);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP force CPU node count: " << webgpu_ep_config.force_cpu_node_names.size();

  //
  // STEP.2 - prepare WebGpuContextConfig
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

  size_t webgpu_device = 0;
  std::string webgpu_device_str;
  if (config_options.TryGetConfigEntry(kWebGpuDevice, webgpu_device_str)) {
    static_assert(sizeof(WGPUDevice) == sizeof(size_t), "WGPUDevice size mismatch");
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(webgpu_device_str.data(), webgpu_device_str.data() + webgpu_device_str.size(), webgpu_device).ec);
  }

  size_t dawn_proc_table = 0;
  std::string dawn_proc_table_str;
  if (config_options.TryGetConfigEntry(kDawnProcTable, dawn_proc_table_str)) {
    ORT_ENFORCE(std::errc{} ==
                std::from_chars(dawn_proc_table_str.data(), dawn_proc_table_str.data() + dawn_proc_table_str.size(), dawn_proc_table).ec);
  }

  webgpu::ValidationMode validation_mode =
#ifndef NDEBUG
      webgpu::ValidationMode::Full  // for debug build, enable full validation by default
#else
      webgpu::ValidationMode::Basic  // for release build, enable basic validation by default
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

  std::string preserve_device_str;
  bool preserve_device = false;
  if (config_options.TryGetConfigEntry(kPreserveDevice, preserve_device_str)) {
    if (preserve_device_str == kPreserveDevice_ON) {
      preserve_device = true;
    } else if (preserve_device_str == kPreserveDevice_OFF) {
      preserve_device = false;
    } else {
      ORT_THROW("Invalid preserve device: ", preserve_device_str);
    }
  }

  webgpu::WebGpuContextConfig context_config{
      context_id,
      reinterpret_cast<WGPUInstance>(webgpu_instance),
      reinterpret_cast<WGPUDevice>(webgpu_device),
      reinterpret_cast<const void*>(dawn_proc_table),
      validation_mode,
      preserve_device,
  };

  LOGS_DEFAULT(VERBOSE) << "WebGPU EP Device ID: " << context_id;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUInstance: " << webgpu_instance;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUDevice: " << webgpu_device;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP DawnProcTable: " << dawn_proc_table;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP ValidationMode: " << validation_mode;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP PreserveDevice: " << preserve_device;

  //
  // STEP.3 - prepare parameters for WebGPU context initialization.
  //

  int backend_type = 0;
#ifdef _WIN32
  // Setup Windows default backend type based on the build configuration
#if defined(DAWN_ENABLE_D3D12)
  backend_type = static_cast<int>(WGPUBackendType_D3D12);
#elif defined(DAWN_ENABLE_VULKAN)
  backend_type = static_cast<int>(WGPUBackendType_Vulkan);
#endif
#endif

  std::string backend_type_str;
  if (config_options.TryGetConfigEntry(kDawnBackendType, backend_type_str)) {
    if (backend_type_str == kDawnBackendType_D3D12) {
      backend_type = static_cast<int>(WGPUBackendType_D3D12);
    } else if (backend_type_str == kDawnBackendType_Vulkan) {
      backend_type = static_cast<int>(WGPUBackendType_Vulkan);
    } else {
      ORT_THROW("Invalid Dawn backend type: ", backend_type_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP Dawn backend type: " << backend_type;

  // buffer cache modes
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

  webgpu::WebGpuBufferCacheConfig buffer_cache_config;

  buffer_cache_config.storage.mode = parse_buffer_cache_mode(kStorageBufferCacheMode,
                                                             webgpu::BufferCacheMode::Bucket);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP storage buffer cache mode: " << buffer_cache_config.storage.mode;

  buffer_cache_config.uniform.mode = parse_buffer_cache_mode(kUniformBufferCacheMode,
                                                             webgpu::BufferCacheMode::Simple);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP uniform buffer cache mode: " << buffer_cache_config.uniform.mode;

  buffer_cache_config.query_resolve.mode = parse_buffer_cache_mode(kQueryResolveBufferCacheMode, webgpu::BufferCacheMode::Disabled);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP query resolve buffer cache mode: " << buffer_cache_config.query_resolve.mode;

  buffer_cache_config.default_entry.mode = parse_buffer_cache_mode(kDefaultBufferCacheMode, webgpu::BufferCacheMode::Disabled);
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP default buffer cache mode: " << buffer_cache_config.default_entry.mode;

  bool enable_pix_capture = false;
  std::string enable_pix_capture_str;
  if (config_options.TryGetConfigEntry(kEnablePIXCapture, enable_pix_capture_str)) {
    if (enable_pix_capture_str == kEnablePIXCapture_ON) {
      enable_pix_capture = true;
    } else if (enable_pix_capture_str == kEnablePIXCapture_OFF) {
      enable_pix_capture = false;
    } else {
      ORT_THROW("Invalid enable pix capture: ", enable_pix_capture_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP pix capture enable: " << enable_pix_capture;

  //
  // STEP.4 - start initialization.
  //

  // Load the Dawn library and create the WebGPU instance.
  auto& context = webgpu::WebGpuContextFactory::CreateContext(context_config);

  // Create WebGPU device and initialize the context.
  context.Initialize(buffer_cache_config, backend_type, enable_pix_capture);

  // Create WebGPU EP factory.
  return std::make_shared<WebGpuProviderFactory>(context_id, context, std::move(webgpu_ep_config));
}

}  // namespace onnxruntime
