// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <charconv>
#include <mutex>

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/providers/webgpu/data_transfer.h"
using namespace onnxruntime::webgpu::options;

namespace onnxruntime {
// Helper struct that holds configuration parameters for creating a WebGPU context with default settings.
// This is used during lazy initialization of the data transfer to create a context if one doesn't exist.
struct WebGpuContextParams {
  webgpu::WebGpuContextConfig context_config;           // WebGPU context configuration
  webgpu::WebGpuBufferCacheConfig buffer_cache_config;  // Buffer cache settings
  int backend_type;                                     // Dawn backend type (D3D12, Vulkan, etc.)
  bool enable_pix_capture;                              // Enable PIX GPU capture for debugging
};

static WebGpuContextParams GetDefaultWebGpuContextParams() {
  WebGpuContextParams params;
  params.context_config.context_id = 0;
  params.context_config.instance = nullptr;
  params.context_config.device = nullptr;
  params.context_config.dawn_proc_table = nullptr;
  params.context_config.validation_mode = webgpu::ValidationMode::Disabled;
  params.context_config.preserve_device = false;
  params.context_config.max_storage_buffer_binding_size = 0;
  params.context_config.power_preference = static_cast<int>(WGPUPowerPreference_HighPerformance);

  params.buffer_cache_config.storage.mode = webgpu::BufferCacheMode::Bucket;
  params.buffer_cache_config.uniform.mode = webgpu::BufferCacheMode::Simple;
  params.buffer_cache_config.query_resolve.mode = webgpu::BufferCacheMode::Disabled;
  params.buffer_cache_config.default_entry.mode = webgpu::BufferCacheMode::Disabled;

#ifdef _WIN32
#if defined(DAWN_ENABLE_D3D12)
  params.backend_type = static_cast<int>(WGPUBackendType_D3D12);
#elif defined(DAWN_ENABLE_VULKAN)
  params.backend_type = static_cast<int>(WGPUBackendType_Vulkan);
#else
  params.backend_type = static_cast<int>(WGPUBackendType_D3D12);
#endif
#else
  params.backend_type = 0;
#endif
  params.enable_pix_capture = false;
  return params;
}

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

  uint64_t max_storage_buffer_binding_size = 0;
  std::string max_storage_buffer_binding_size_str;
  if (config_options.TryGetConfigEntry(kMaxStorageBufferBindingSize, max_storage_buffer_binding_size_str)) {
    ORT_ENFORCE(
        std::errc{} == std::from_chars(
                           max_storage_buffer_binding_size_str.data(),
                           max_storage_buffer_binding_size_str.data() + max_storage_buffer_binding_size_str.size(),
                           max_storage_buffer_binding_size)
                           .ec,
        "Invalid maxStorageBufferBindingSize value: ", max_storage_buffer_binding_size_str);
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP max storage buffer binding size: " << max_storage_buffer_binding_size;

  // power preference
  int power_preference = static_cast<int>(WGPUPowerPreference_HighPerformance);  // default
  std::string power_preference_str;
  if (config_options.TryGetConfigEntry(kPowerPreference, power_preference_str)) {
    if (power_preference_str == kPowerPreference_HighPerformance) {
      power_preference = static_cast<int>(WGPUPowerPreference_HighPerformance);
    } else if (power_preference_str == kPowerPreference_LowPower) {
      power_preference = static_cast<int>(WGPUPowerPreference_LowPower);
    } else {
      ORT_THROW("Invalid power preference: ", power_preference_str);
    }
  }
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP power preference: " << power_preference;

  webgpu::WebGpuContextConfig context_config{
      context_id,
      reinterpret_cast<WGPUInstance>(webgpu_instance),
      reinterpret_cast<WGPUDevice>(webgpu_device),
      reinterpret_cast<const void*>(dawn_proc_table),
      validation_mode,
      preserve_device,
      max_storage_buffer_binding_size,
      power_preference,
  };

  LOGS_DEFAULT(VERBOSE) << "WebGPU EP Device ID: " << context_id;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUInstance: " << webgpu_instance;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUDevice: " << webgpu_device;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP DawnProcTable: " << dawn_proc_table;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP ValidationMode: " << validation_mode;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP PreserveDevice: " << preserve_device;
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP PowerPreference: " << power_preference;

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

// WebGPU DataTransfer implementation wrapper for the C API with lazy initialization
struct WebGpuDataTransferImpl : OrtDataTransferImpl {
  WebGpuDataTransferImpl(const OrtApi& ort_api_in)
      : ort_api{ort_api_in},
        ep_api{*ort_api_in.GetEpApi()},
        data_transfer_{nullptr},
        context_id_{0},  // Always use context 0 for Environment's data transfer
        init_mutex_{} {
    ort_version_supported = ORT_API_VERSION;
    CanCopy = CanCopyImpl;          // OrtDataTransferImpl::CanCopy callback
    CopyTensors = CopyTensorsImpl;  // OrtDataTransferImpl::CopyTensors callback
    Release = ReleaseImpl;          // OrtDataTransferImpl::Release callback
  }

  static bool CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                          const OrtMemoryDevice* src_memory_device,
                          const OrtMemoryDevice* dst_memory_device) noexcept {
    const auto& impl = *static_cast<const WebGpuDataTransferImpl*>(this_ptr);
    OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);

    // Check if at least one device is GPU
    bool has_gpu = (src_type == OrtMemoryInfoDeviceType_GPU) || (dst_type == OrtMemoryInfoDeviceType_GPU);
    if (!has_gpu) {
      return false;
    }

    // WebGPU uses vendor ID 0 (VendorIds::NONE). Only handle GPU devices with vendor ID 0.
    // This prevents attempting to copy data for other EPs' fake GPU devices (e.g., example EP with vendor 0xBE57)
    if (src_type == OrtMemoryInfoDeviceType_GPU) {
      uint32_t src_vendor = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
      if (src_vendor != 0) {
        return false;  // Not a WebGPU device
      }
    }

    if (dst_type == OrtMemoryInfoDeviceType_GPU) {
      uint32_t dst_vendor = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);
      if (dst_vendor != 0) {
        return false;  // Not a WebGPU device
      }
    }

    // If both are GPU, they must have the same device ID
    if (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) {
      uint64_t src_device_id = impl.ep_api.MemoryDevice_GetDeviceId(src_memory_device);
      uint64_t dst_device_id = impl.ep_api.MemoryDevice_GetDeviceId(dst_memory_device);
      if (src_device_id != dst_device_id) {
        return false;  // Cannot copy between different devices
      }
    }

    // WebGPU supports GPU<->GPU, GPU<->CPU copies (where GPU has vendor ID 0)
    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
           (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
           (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
  }

  static OrtStatus* CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                    const OrtValue** src_tensors,
                                    OrtValue** dst_tensors,
                                    OrtSyncStream** /*streams*/,
                                    size_t num_tensors) noexcept {
    auto& impl = *static_cast<WebGpuDataTransferImpl*>(this_ptr);

    if (num_tensors == 0) {
      return nullptr;
    }

    // Lazy initialization: Use double-checked locking to avoid unnecessary lock operations
    if (impl.data_transfer_ == nullptr) {
      std::lock_guard<std::mutex> lock(impl.init_mutex_);
      if (impl.data_transfer_ == nullptr) {
        // Always create a new context with context_id 0
        WebGpuContextParams params = GetDefaultWebGpuContextParams();
        params.context_config.context_id = impl.context_id_;
        auto* context_ptr = &webgpu::WebGpuContextFactory::CreateContext(params.context_config);
        context_ptr->Initialize(params.buffer_cache_config, params.backend_type, params.enable_pix_capture);

        // Create the DataTransfer instance
        // Note: The DataTransfer holds a const reference to BufferManager. The BufferManager's lifecycle
        // is managed by the WebGpuContext, which is stored in a static WebGpuContextFactory and persists
        // for the lifetime of the application, ensuring the reference remains valid.
        impl.data_transfer_ = std::make_unique<webgpu::DataTransfer>(context_ptr->BufferManager());
      }
    }

    // Now perform the actual tensor copy
    for (size_t idx = 0; idx < num_tensors; ++idx) {
      const OrtValue* src_tensor = src_tensors[idx];
      OrtValue* dst_tensor = dst_tensors[idx];
      auto status = impl.data_transfer_->CopyTensor(src_tensor->Get<Tensor>(), *dst_tensor->GetMutable<Tensor>());
      if (!status.IsOK()) {
        return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, status.ErrorMessage().c_str());
      }
    }
    return nullptr;
  }

  static void ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
    auto* p_impl = static_cast<WebGpuDataTransferImpl*>(this_ptr);
    int context_id = p_impl->context_id_;
    bool data_transfer_initialized = false;
    {
      std::lock_guard<std::mutex> lock(p_impl->init_mutex_);
      data_transfer_initialized = (p_impl->data_transfer_ != nullptr);
    }
    delete p_impl;
    if (data_transfer_initialized) {
      webgpu::WebGpuContextFactory::ReleaseContext(context_id);
    }
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  std::unique_ptr<webgpu::DataTransfer> data_transfer_;  // Lazy-initialized
  int context_id_;                                       // Track which context we're using
  std::mutex init_mutex_;                                // Protects lazy initialization
};

OrtDataTransferImpl* OrtWebGpuCreateDataTransfer() {
  // Validate API version is supported
  const OrtApi* api = OrtApis::GetApi(ORT_API_VERSION);
  if (!api) {
    // API version not supported - return nullptr to indicate failure
    return nullptr;
  }
  return new WebGpuDataTransferImpl(*api);
}

}  // namespace onnxruntime
